"""X24 — THE PRIOR-FRAGILITY CENSUS (T285's discriminating observation; Law
3's gate, round 4; dispatched 2026-10-09). This docstring carries the
question, the design, the panel, the bars and P-x24a VERBATIM from the
dispatch letter, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (verbatim): "is a name's lift under a fixed full-dose generic
displacement a generic function of its prior (or anchoring), or
name-specific structure?"

BACKGROUND (dispatch, verbatim in substance): x23 found the complement-read
effect follows the NAME SLOT, not the write — a full-dose out-of-room
complement lifts the dead TAVIREN name (count-0, ~0.004 prior) ~7x over
prior regardless of which write the complement came from, while the formed
host name Z is unmoved. Mechanism candidate: PRIOR-FRAGILITY — weakly
anchored name logits drift up under large generic displacement; strongly
anchored ones do not. If lift-vs-prior is one generic curve across the
prior spectrum, the necessity law RESTORES with a prior-relative read-bar
rider (absolute 0.05 bars are name-confounded).

THE DESIGN (dispatch, verbatim in substance):
  "1. ONE fixed displacement: the K10K write's complement at full dose
  (x15's exact construction — rebuild by its convention, md5-bind; it is the
  displacement whose effect on two names is already committed: Z 0.00073
  unmoved, TAVIREN 0.0273 lifted).
  2. THE NAME PANEL (at the host's 60 g0 contexts, and at a neutral corpus
  context set as a second reading site): (i) the formed host name Z
  (anchor); (ii) TAVIREN (count-0 synthetic, committed reference 0.0273);
  (iii) 2-3 FRESH count-0 synthetic names from the committed name banks
  (e.g., e293's family-B bank — verify on disk; construct with the same
  count-0/prior gate convention e311 used); (iv) 3-4 corpus names across
  the prior spectrum (a very common token, a mid-frequency proper name, a
  rare-but-seen name — select from the corpus with stated frequency
  counts); (v) a scrambled-string control (zero prior, no corpus presence).
  3. Measure per name: prior p(name) at baseline (no displacement), p(name)
  under the displacement, the lift ratio; plus the full-write control on Z
  and TAVIREN (committed; reproduce).
  4. Report the LIFT-vs-PRIOR scatter across all names at both reading
  sites."

THE PANEL (frozen at birth; every corpus count stated, train split,
design-time string counts — the e293 selection-probe precedent):
  * ANCHOR  ZEPHYRA (initial Z) — "the formed host name Z": the K10K
    write's own target name (G1.NAME verbatim), formed 0.2646-class by the
    full write, dead 1.34e-5-class at base. Its committed complement read
    0.0007308 (x15's KK diagonal).
  * TAVIREN (T) — count-0 synthetic, e293's committed family-B name;
    committed references: prior 0.0023043, complement read 0.0272878 (x23's
    KT cell).
  * FRESH BANK: QELVARO (Q), BUVONDI (B), NYSTORA (N) — the three e293
    NAME_BANK members not already spent by the x-series (ZEPHYRA is the
    anchor's own name, TAVIREN is the committed reference); each gated by
    e311's convention (7 letters, every char in the 65-char vocab, count-0
    in the train split, != G1.NAME).
  * SCRAMBLED CONTROL: VIRETAN (V) — a fixed permutation of TAVIREN's
    letters (the same multiset, initial V, count-0, distinct from every
    bank name).
  * CORPUS NAMES (train-split counts stated, frozen exact-equality):
    MAMILLIUS (M, 13 — rare-but-seen), LEONTES (L, 125 — mid-frequency
    proper name), KING (K, 556 — very common token), and the two host
    corpus names ELIZABETH (E, 105) / FLORIZEL (F, 45) — the high-prior
    formed class at the host site by construction.
  11 names, 11 DISTINCT initial chars (Z T Q B N V M L K E F) — every name
  occupies its own name slot at every read.

THE TWO READING SITES (frozen):
  * HOST-G0: the 60 host g0 install contexts (e261's splice bank verbatim:
  corpus seed 1337, find_occ p >= 280, SPLICE_RNG 24301 shuffle, first 60,
  mix FLORIZEL 19 / ELIZABETH 41, PRE 130, shape [60,130]) — the exact bank
  on which the committed KK/KT references were read.
  * NEUTRAL: 60 fresh windows from the TRAIN split, each 130 chars ending
  at an uppercase ASCII letter (a name-slot-like position), with no HOST
  string (FLORIZEL/ELIZABETH) inside the window or the following 30 chars;
  candidates shuffled with random.Random(24001), first 60. Registered seed
  24001; construction deterministic.

BARS (frozen VERBATIM from the dispatch, BEFORE any compute):
  - ONE-GENERIC-CURVE: "the lift ratios of all count-0/low-prior names
    (synthetic + rare corpus) fall within ~3x of each other AND the formed
    anchor's lift is <= 0.2x theirs AND the mid/high-prior corpus names'
    lifts fall between — prior-fragility is generic; NECESSITY RESTORES
    with the rider 'reads are scored prior-relative (a complement floors
    below ~10x prior; a full write clears it)'; tonight's write-dependence
    scare re-closes as an instrument lesson."
  - NAME-SPECIFIC-STRUCTURE: "some marginal names lift >> others (>10x
    spread among same-prior names) — fragility is structure, not prior; the
    census continues with name-geometry probes; necessity stays weakened."

REGISTERED PREDICTION P-x24a (frozen VERBATIM from the dispatch, BEFORE any
compute): "Register P-x24a BEFORE compute. Lab guess: ONE-GENERIC-CURVE."

OPERATIONALIZATIONS (frozen HERE at birth BEFORE compute; they fix the
clauses, they do not move the bars):
  * THE READ CURRENCY (the series' convention, x15/x23/e311): a name's read
    := p(name[0]) — the name-INITIAL char's probability at the final
    context position, mean over the site's 60 contexts. The committed
    references (KK 0.0007308, KT 0.0272878, priors 1.3384e-05 / 0.0023043)
    are in exactly this currency.
  * LIFT (the dispatch's own defined measure — "Measure per name: prior
    p(name) at baseline, p(name) under the displacement, the lift ratio"):
    lift(i,S) := p_comp(i,S) / max(p_base(i,S), 1e-12). The SAME ratio
    currency is used in EVERY clause, including the anchor clause. Absolute
    gains (p_comp - p_base), logit deltas, and the anchor's formed-fraction
    (p_comp(Z)/p_fullK(Z)) are co-reported, never adjudicated.
  * DISCLOSED PRE-COMPUTE TENSION (registered, not shopped): the committed
    values already give lift(ZEPHYRA) = 0.0007308/1.3384e-05 = 54.6x and
    lift(TAVIREN) = 0.0272878/0.0023043 = 11.8x at the host site — the
    anchor clause (anchor <= 0.2x theirs) is strained by committed data
    BEFORE this census runs. The bars are frozen anyway; the verdict is
    what it is.
  * CLAUSE COHORTS (frozen by CONSTRUCTION, not by measured prior):
    cohort-A "count-0/low-prior names (synthetic + rare corpus)" :=
    TAVIREN, QELVARO, BUVONDI, NYSTORA, VIRETAN, MAMILLIUS. The anchor
    ZEPHYRA is EXCLUDED from cohort-A (the bars give it its own clause);
    disclosed confound: the anchor is also the displacement's parent
    write's own target name, so its lift may carry its own write's
    out-of-room remainder — the fresh bank names are the write-independent
    cohort. cohort-C "mid/high-prior corpus names" := KING, LEONTES,
    ELIZABETH, FLORIZEL.
  * CLAUSE FORMS (frozen): per site S —
    A(S) := max(lift over cohort-A at S) / min(lift over cohort-A at S)
           <= 3.0;
    B(S) := lift(anchor, S) <= 0.2 * median(lift over cohort-A at S);
    C(S) := every cohort-C lift at S lies inside the closed interval
            [min(anchor lift, cohort-A lifts), max(anchor lift, cohort-A
            lifts)] at S.
    ONE-GENERIC-CURVE := A(host) AND B(host) AND C(host) AND A(neutral)
    AND B(neutral) AND C(neutral) ("ALL ... names", both reading sites).
  * NAME-SPECIFIC-STRUCTURE := there exists a pair of distinct panel names
    at the same site whose base priors are within 3x of each other (both
    directions) AND whose lift ratio (max/min) exceeds 10x. Precedence:
    ONE-GENERIC-CURVE first; then NAME-SPECIFIC-STRUCTURE; neither => GAP
    (verbatim report; no wording change without a new registered cell).
  * PRIOR-GATE CONVENTION (e311's, ported): fresh synthetics + scrambled +
    TAVIREN — 7 letters, every char in vocab, count-0 in the train split,
    != G1.NAME; bank-discipline band (e293's): base prior at host-g0
    <= 0.004 co-reported per name; a straddle in (0.004, 0.05] is a
    DISCLOSED DEVIATION (no spare bank members exist: the 5-name bank has
    ZEPHYRA/TAVIREN spent); a base prior > 0.05 (e311's G_BASE bar) HALTS
    => TEXTURE.
  * STATES (one applied state, many probes): base := e001 (md5-bound);
    comp := THE displacement, the K10K write's complement at full dose —
    x15's construction rebuilt bit-exact AND verified bit-equal to the
    committed carrier runs/x15/x15_comp_xFULL.pt, then probed FROM THE
    CARRIER'S OWN model dict (one loaded net for every name read); fullK
    := base + dW_k10k in-memory (x15's exact G_FULLREAD path) — the
    committed full-write control on Z; fullT := the TAVIREN full write,
    probed from x17's committed carrier runs/x17/x17_x15R_fullwrite_control.pt
    ["model"] (md5-bound) — the committed full-write control on TAVIREN.
    fullK/fullT are additionally read on the WHOLE panel (the "full write
    clears ~10x prior" rider's evidence, co-reported, never a bar).
  * GATES (a failure HALTS => TEXTURE, nothing adjudicated): G_ENDPOINTS
    (10 md5 binds incl. both carriers + the e293 bank source + committed
    references read from artifacts, never retyped), G_FLATBASIS, G_PANEL
    (+G_NAMEFREE), G_BATTERY, G_NEUTRAL, G_BASEPRIOR, G_PANELPRIOR, G_ROOM,
    G_ORTH, G_REGEN, G_DOSEMATCH, G_INROOM, G_FULLREAD (both full-write
    controls), G_DIAG (both committed complement references reproduced,
    x15's [0.5x, 2x] band).
  * HONESTY RIDERS (never bars): ce_r per state (x17's G1.ce_fixed_cpu),
    per (state, site): mean top-1 p, mean entropy; per name: median,
    frac >= 0.05, frac argmax; Spearman rho of log10(prior) vs log10(lift)
    over the full panel per site (the monotone-generic-curve co-test); the
    anchor's formed-fraction; the 10x-prior rider census (which names show
    comp lift < 10x AND fullK lift > 10x).
  * REGISTERED SEEDS (this cell's, frozen): global 24000; neutral bank
    24001; room cert 24005. No fresh random draws exist in this design —
    every vector is deterministic from its md5-bound parents.
  * CPU ENVELOPE (dispatch): CUDA_VISIBLE_DEVICES="" BEFORE torch import;
    torch threads 4; pocketfft workers 4; NO GPU, no envelope-log writes,
    no other runs/ touched (write ONLY runs/x24*); timestamps
    datetime.now(UTC) only.
  * Outputs: runs/x24/{metrics.json (PROGRESSIVE), REPORT.md,
    x24_prior_census.png} (the lift-vs-prior scatter at both reading sites,
    the committed Z/T points drawn as references). NO new .pt artifacts
    (everything probed from committed carriers or in-memory from bound
    parents — the *.pt gitignore convention). NO NOTES/THINKING/QUEUE/
    STATE edits (dispatch — the heartbeat folds). Birth commit BEFORE
    compute; smoke pass disclosed; final commit AND push.
  * Smoke (X24_SMOKE=1): the FULL gate path LIVE (parents + room +
    complement + carrier + panel + both sites + all states + all reads);
    own smoke dir runs/x24_smoke/; NO adjudication, NO figure, NO report
    (SMOKE stamp on every read; nothing adjudicated).

COMPUTE: ~4 fp64 DCT projections + 8 site-state forward passes + 4 ce_r
evals on the 2.74M organism — minutes, CPU-only.

Run:  python lab/x24_prior_census.py          (X24_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-ONLY, bulletproof

import hashlib                                   # noqa: E402
import json                                      # noqa: E402
import math                                      # noqa: E402
import random                                    # noqa: E402
import re as _re                                # noqa: E402
import subprocess                                # noqa: E402
import sys                                       # noqa: E402
import time                                      # noqa: E402
from datetime import datetime, timezone          # noqa: E402
from pathlib import Path                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")     # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                             # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                # noqa: BLE001
    pass

import numpy as np                               # noqa: E402
import torch                                     # noqa: E402
import torch.nn.functional as F                  # noqa: E402

import common                                    # noqa: E402
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                       # noqa: E402 (REPO,
                                                  # find_occ, SPLICE_RNG)
import g1b_continuity as GB                      # noqa: E402 — MUST be
                                                  # imported BEFORE G1
import g1_anchored_ball as G1                    # noqa: E402
import e261_rank_ladder as E261                  # noqa: E402
from e261_rank_ladder import SRCT                # noqa: E402

torch.set_num_threads(4)          # CPU-only cell; the shared desk lane
E261.DCT_WORKERS = 4              # x15/x17/x23's desk convention (disclosed)

SMOKE = os.environ.get("X24_SMOKE") == "1"
NAME = "x24_smoke" if SMOKE else "x24"

T0 = time.time()
RD = run_dir(NAME)
REPO = E43.REPO
CKPT_DIR = GB.CKPT_DIR


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(m: str) -> None:
    print(f"[x24 {time.time() - T0:7.1f}s] {m}", flush=True)


def md5of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                    # noqa: BLE001
        return "unavailable"


# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
# ---- the shared substrate -------------------------------------------------
BASE_CK = "e001.pt"
BASE_MD5 = "d114536d1c0983ab3be67f67ff0667c8"
FACT_CK = "e261_K10K_inst_resume.pt"            # the K10K write's end state
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
ROOMS264_CK = "e264_rooms.pt"
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"
ROOM_K = 10_000
ROOM_SEED_D, ROOM_SEED_S = 26113, 26114
N_PARAMS = GB.G1B_PARAMS                          # 2,739,072
GLOBAL_SEED = 24000
NEUTRAL_SEED = 24001
CERT_SEED = 24005

# ---- the parents' committed artifacts (md5-bound) --------------------------
X15_METRICS = REPO / "runs" / "x15" / "metrics.json"
X15_METRICS_MD5 = "c90dd9371a5a7e248fdbd92c06bb243a"
X15_COMP_CARRIER = REPO / "runs" / "x15" / "x15_comp_xFULL.pt"
X15_COMP_CARRIER_MD5 = "b0a1785c6f759fdd1d8deeae7fee4a9b"
X23_METRICS = REPO / "runs" / "x23" / "metrics.json"
X23_METRICS_MD5 = "cb8a36f5e53615827aecfa1680155e44"
X17_METRICS = REPO / "runs" / "x17" / "metrics.json"
X17_METRICS_MD5 = "98a6138279852ca0705718d77a9c28f7"
X17_FULL_CARRIER = REPO / "runs" / "x17" / "x17_x15R_fullwrite_control.pt"
X17_FULL_CARRIER_MD5 = "c0a2a9574ebf0b71094ae5e254cc965f"
E311_METRICS = REPO / "runs" / "e311" / "metrics.json"
E311_METRICS_MD5 = "4efbb2242c8e2e3abab1a2cb63e3e35a"
E293_SOURCE = REPO / "lab" / "e293_distinct_contention.py"   # the bank
E293_SOURCE_MD5 = "1f3e16968359171d3db1a52824c48cf4"

# ---- frozen literals (cross-checked against the md5-bound artifacts) ------
# the K10K write (x15/e290/e310's committed object)
K10K_WRITE_NORM = 9.1788432658723
K10K_FULL_G0_Z = 0.26464763283729553             # the full write on Z (host)
K10K_BASE_G0_Z = 1.3383959412749391e-05          # base prior, Z, host g0
X15_S_FULL = 3.0349885643664365                  # x15's committed scalar
X15_COMP_L2_64 = 3.024341960836486               # natural complement L2
X15_DIAG_KK = 0.0007308000349439681              # THE committed Z reference
# the TAVIREN name (e311/x17/x23's committed object)
X23_KT = 0.027287840843200684                    # THE committed T reference
X17_BASE_G0_T = 0.0023042955435812473            # base prior, T, host g0
E311_FRESH_G0_T = 0.285851389169693              # fullT on T (host g0)
E293_BANK = ("ZEPHYRA", "TAVIREN", "QELVARO", "BUVONDI", "NYSTORA")

# ---- the panel (frozen; corpus counts = train split, exact-equality) ------
PANEL = [
    # (name, class, cohort, train_count_expected)
    ("ZEPHYRA",   "anchor_formed_host",  "anchor", 0),
    ("TAVIREN",   "synthetic_count0",    "A",      0),
    ("QELVARO",   "synthetic_count0_fresh_bank", "A", 0),
    ("BUVONDI",   "synthetic_count0_fresh_bank", "A", 0),
    ("NYSTORA",   "synthetic_count0_fresh_bank", "A", 0),
    ("VIRETAN",   "scrambled_control",   "A",      0),
    ("MAMILLIUS", "corpus_rare_seen",    "A",      13),
    ("LEONTES",   "corpus_mid_proper",   "C",      125),
    ("KING",      "corpus_common_token", "C",      556),
    ("ELIZABETH", "corpus_host_high",    "C",      105),
    ("FLORIZEL",  "corpus_host_high",    "C",      45),
]
COHORT_A = [n for n, _, c, _ in PANEL if c == "A"]
COHORT_C = [n for n, _, c, _ in PANEL if c == "C"]
ANCHOR = "ZEPHYRA"

# ---- the bars (frozen verbatim; see docstring) -----------------------------
SPREAD_BAR = 3.0        # clause A "within ~3x"
ANCHOR_FRAC = 0.2       # clause B "<= 0.2x theirs"
SAME_PRIOR_X = 3.0      # structure bar: same-prior := within 3x in prior
STRUCT_SPREAD = 10.0    # structure bar: ">10x spread"
REPRO_BAND = (0.5, 2.0)   # x15's G_COMPREF convention
FULLREAD_TOL = 5e-3        # x15's own G_FULLREAD bar
PRIOR_FLOOR = 1e-12        # lift denominator floor (disclosed)
BANK_BAND = 0.004          # e293's bank-discipline band (co-report)
BASE_BAR = 0.05            # e311's G_BASE bar (halt)

REGISTERED = {
    "question_verbatim":
        "is a name's lift under a fixed full-dose generic displacement a "
        "generic function of its prior (or anchoring), or name-specific "
        "structure?",
    "bars_verbatim": {
        "ONE-GENERIC-CURVE":
            "the lift ratios of all count-0/low-prior names (synthetic + "
            "rare corpus) fall within ~3x of each other AND the formed "
            "anchor's lift is <= 0.2x theirs AND the mid/high-prior corpus "
            "names' lifts fall between — prior-fragility is generic; "
            "NECESSITY RESTORES with the rider 'reads are scored "
            "prior-relative (a complement floors below ~10x prior; a full "
            "write clears it)'; tonight's write-dependence scare re-closes "
            "as an instrument lesson.",
        "NAME-SPECIFIC-STRUCTURE":
            "some marginal names lift >> others (>10x spread among "
            "same-prior names) — fragility is structure, not prior; the "
            "census continues with name-geometry probes; necessity stays "
            "weakened.",
    },
    "P_x24a_verbatim": "Register P-x24a BEFORE compute. Lab guess: "
                       "ONE-GENERIC-CURVE.",
    "executor_position":
        "Pre-compute, registered: the strict conjunction is strained by "
        "committed data BEFORE this census runs — lift(ZEPHYRA) at host = "
        "0.0007308/1.3384e-05 = 54.6x vs lift(TAVIREN) = 0.0272878/"
        "0.0023043 = 11.8x, and clause B needs the anchor <= 0.2x the "
        "cohort median. Executor sub-prediction P-x24a-exec (scored): "
        "ONE-GENERIC-CURVE does NOT fire (clause B fails at the host "
        "site); NAME-SPECIFIC-STRUCTURE does not fire either (no same-"
        "prior pair splits >10x); the verdict is GAP with the monotone "
        "rider (Spearman rho of log-prior vs log-lift < 0 on the pooled "
        "panel at BOTH sites) — prior-fragility generic in trend, "
        "name-slot-confounded absolute bars for the fold to re-register.",
    "clauses_fixed": {
        "read_currency": "a name's read := p(name[0]) at the final context "
                         "position, mean over the site's 60 contexts "
                         "(x15/x23/e311's committed currency)",
        "lift": "lift(i,S) := p_comp(i,S) / max(p_base(i,S), 1e-12); the "
                "same ratio currency in EVERY clause incl. the anchor "
                "clause; absolute gains, logit deltas and formed-fraction "
                "co-reported only",
        "cohorts": "cohort-A := TAVIREN, QELVARO, BUVONDI, NYSTORA, "
                   "VIRETAN, MAMILLIUS (synthetic + rare corpus; the "
                   "anchor excluded — its own clause); cohort-C := KING, "
                   "LEONTES, ELIZABETH, FLORIZEL; frozen by construction",
        "clause_A": "max/min lift over cohort-A <= 3.0, per site",
        "clause_B": "lift(anchor) <= 0.2 * median(cohort-A lifts), per "
                    "site",
        "clause_C": "every cohort-C lift inside [min(anchor, cohort-A "
                    "lifts), max(anchor, cohort-A lifts)], per site",
        "one_generic_curve": "A AND B AND C at BOTH sites",
        "name_specific": "a same-site pair of distinct panel names with "
                         "base priors within 3x both ways AND lift "
                         "max/min > 10x",
        "precedence": "ONE-GENERIC-CURVE first; then NAME-SPECIFIC-"
                      "STRUCTURE; neither => GAP (verbatim report; no "
                      "wording change without a new registered cell)",
    },
    "registration": "bars + P-x24a VERBATIM from the dispatch letter; "
                    "operationalizations frozen HERE at birth BEFORE "
                    "compute; this script committed at birth; adjudicate "
                    "against exactly this; no bar shopping.",
}

DEVIATIONS = [
    "CPU-ONLY cell (dispatch: another agent owns the GPU lane) — "
    "CUDA_VISIBLE_DEVICES='' before torch import, torch threads 4, "
    "pocketfft workers 4, no GPU code path, no envelope-log writes, no "
    "other runs/ touched.",
    "THE READ CURRENCY: a name's read is its INITIAL char's probability "
    "at the final context position (the committed KK/KT currency); "
    "multi-char name bodies are not read (disclosed; the series' "
    "name-slot convention).",
    "CLAUSE CURRENCIES: lift = ratio p_comp/p_base for every clause "
    "(the dispatch's own defined measure); the committed values already "
    "strain clause B at the host site (54.6x vs 11.8x) — registered "
    "pre-compute, disclosed in REGISTERED.executor_position.",
    "The anchor ZEPHYRA is the displacement's parent write's own target "
    "name — a disclosed confound (its lift may carry its own write's "
    "out-of-room remainder); the fresh bank names are the "
    "write-independent cohort.",
    "Cohort membership frozen by CONSTRUCTION; measured priors co-"
    "reported per site (KING/LEONTES may measure low-prior at the host "
    "site — the site-dependence is a finding, not a reclassification).",
    "Bank-discipline band (e293, <= 0.004 at the name slot) co-reported; "
    "halt only above e311's 0.05 G_BASE bar; no spare bank members exist "
    "(5-name bank, ZEPHYRA/TAVIREN spent), so a straddle in (0.004, "
    "0.05] is disclosed, not swapped.",
    "NEUTRAL site rule frozen: 60 windows, 130 chars, ending at an "
    "uppercase ASCII letter, host-free in window + 30-char lookahead, "
    "train split, random.Random(24001) shuffle, first 60.",
    "The scrambled control VIRETAN is a fixed permutation of TAVIREN's "
    "letters (initial V) — zero corpus presence gated at runtime.",
    "gm12 NOT read (the dispatch froze the site axis at host-g0 + "
    "neutral); fullK/fullT read on the whole panel as rider evidence for "
    "the 10x-prior restoration rider (never a bar).",
    "No fresh random draws in the adjudicated design (deterministic from "
    "md5-bound parents + frozen seeds 24000/24001/24005); n=1 per cell "
    "(one lineage, one session — the g-series standing lottery note).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).",
    "Smoke (X24_SMOKE=1): full gate path + all reads live, own smoke dir; "
    "NOTHING adjudicated.",
]

METRICS: dict = {
    "experiment": "x24_prior_census",
    "phase": "THE PRIOR-FRAGILITY CENSUS (T285): one fixed full-dose "
             "generic displacement (the K10K write's complement, x15's "
             "construction, carrier-bound) x an 11-name panel across the "
             "prior spectrum x two reading sites (host g0 + neutral) — "
             "is lift a generic function of prior, or name-specific "
             "structure?",
    "date": now_utc(),
    "status": "PARTIAL: startup (bars + P-x24a registered at birth)",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": 4},
    "envelope": {
        "device": "CPU ONLY (CUDA_VISIBLE_DEVICES=''; another agent owns "
                  "the GPU lane; no envelope-log writes)",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": DEVIATIONS,
    "builds_on": [
        "x23 (THE CROSS-BATTERY: the KT cell 0.0272878 — the committed "
        "TAVIREN-under-K10K-complement reference this census reproduces; "
        "the battery/read harness this script ports)",
        "x15 (THE K10K dose-matched complement: the construction, the "
        "carrier runs/x15/x15_comp_xFULL.pt, the committed KK diagonal "
        "0.0007308, the G_FULLREAD/G_COMPREF conventions)",
        "T285 (the THINKING card that named prior-fragility and this "
        "census as its discriminating observation)",
        "e311 (the TAVIREN name's certified conventions + the "
        "fullT control reference 0.285851)",
        "e293 (the committed NAME_BANK — ZEPHYRA/TAVIREN/QELVARO/"
        "BUVONDI/NYSTORA — and the count-0/prior-gate bank discipline)",
        "x17 (the fullwrite control carrier for fullT)",
        "e261/e264/e290/e310 (the K10K write, the room, the write norm, "
        "the complement construction the parents ported)",
    ],
    "whats_new": [
        "THE FIRST LIFT-vs-PRIOR SCATTER: one fixed displacement read on "
        "an 11-name panel spanning ~5 decades of prior (anchor, committed "
        "reference, 3 fresh bank names, a scrambled control, 5 corpus "
        "names with stated counts) — the prior-fragility mechanism "
        "candidate's direct test",
        "the SECOND READING SITE (a neutral host-free corpus bank) — the "
        "census is instrument-geometry-doubled (Rule 12): the same names "
        "re-ranked by a site that owes nothing to either write",
        "the fresh-bank cohort (QELVARO/BUVONDI/NYSTORA) — count-0 names "
        "with NO relation to the displacement's parent write, the clean "
        "prior-fragility test T285 lacked",
        "the full-write lifts on the whole panel (the x10-prior "
        "restoration rider's first direct evidence, co-reported)",
    ],
    "gates": {},
}
METRICS_PATH = RD / "metrics.json"


def write_partial(note: str) -> None:
    METRICS["phase_note"] = note
    METRICS["date_updated"] = now_utc()
    save_json(METRICS_PATH, METRICS)
    log(f"[metrics] partial saved ({note})")


# ------------------------------------------------------------------ helpers
def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


@torch.no_grad()
def site_read(net, ids: torch.Tensor, bs: int = 30) -> dict:
    """One forward pass over a site's 60 contexts; keeps the full final
    softmax row-matrix so EVERY name column is read from the SAME pass
    (x23's battery_read numerics, restructured one-pass-many-probes; the
    committed Z/T columns are bit-identical to x23's batching)."""
    net.eval()
    rows, tops, ents = [], [], []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        rows.append(pr)
        tops.append(pr.max(-1).values)
        ents.append(-(pr * torch.log(pr.clamp_min(1e-30))).sum(-1))
    probs = torch.cat(rows)                       # [60, vocab]
    return {"probs": probs,
            "rider_mean_top1_p": float(torch.cat(tops).mean()),
            "rider_mean_entropy": float(torch.cat(ents).mean())}


def name_read(site: dict, cid: int) -> dict:
    """A name's read from a site pass (the x23 battery_read statistics)."""
    p = site["probs"][:, cid]
    lg = torch.log(p.clamp_min(1e-30))
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_pz_ge_0.05": float((p >= 0.05).float().mean()),
            "frac_argmax_z": float((site["probs"].argmax(-1) == cid)
                                   .float().mean()),
            "mean_log_pz": float(lg.mean())}


def inroom_frac(room: SRCT, v64: np.ndarray) -> float:
    vn = float(np.linalg.norm(v64))
    if vn == 0.0:
        return 0.0
    return float(np.linalg.norm(room.project(v64)) / vn)


def inroom_energy_share(room: SRCT, v64: np.ndarray) -> float:
    """x15's G_INROOM_SCALED exact form: ||Pv||^2 / ||v||^2."""
    e = float(v64 @ v64)
    if e == 0.0:
        return 0.0
    p = room.project(v64)
    return float((p @ p) / e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-30), 1.0 - 1e-30)
    return float(math.log(p / (1.0 - p)))


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y) -> float:
    rx, ry = _ranks(list(x)), _ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def texture(halt: str) -> SystemExit:
    METRICS["verdict"] = {"word": "TEXTURE",
                          "why": f"{halt} — nothing adjudicated"}
    write_partial(f"HALT: TEXTURE ({halt})")
    return SystemExit(f"{halt} — nothing adjudicated")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X24 — THE PRIOR-FRAGILITY CENSUS (smoke={SMOKE}) -> {RD}")
    METRICS["birth_commit"] = git_head()
    write_partial("startup (bars + P-x24a registered, committed at birth)")
    set_seed(GLOBAL_SEED)          # global init only; no fresh draws exist

    # ============ P0: corpus, panel gates, the two sites ================
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

    # G_NAMEFREE + G_PANEL (e311's convention, ported + extended)
    e293_src = E293_SOURCE.read_text(encoding="utf-8")
    m = _re.search(r"NAME_BANK\s*=\s*\(([^)]*)\)", e293_src)
    bank_on_disk = tuple(s.strip().strip('"\'') for s in
                         m.group(1).split(",")) if m else ()
    counts = {nm: train_text.count(nm) for nm, *_ in PANEL}
    initials = [nm[0] for nm, *_ in PANEL]
    fresh_bank = [nm for nm, cls, _, _ in PANEL
                  if cls == "synthetic_count0_fresh_bank"]
    scrambled = next(nm for nm, cls, _, _ in PANEL
                     if cls == "scrambled_control")
    synth_all = [nm for nm, cls, _, _ in PANEL
                 if cls.startswith("synthetic") or cls == "scrambled_control"]
    gpanel = {
        "form": "e311's gate convention ported: every synthetic/scrambled "
                "name — 7 letters, every char in the 65-char vocab, "
                "count-0 train split, != G1.NAME; corpus names — exact "
                "frozen counts; initials all distinct; the bank parsed "
                "from e293's md5-bound source",
        "e293_bank_on_disk": list(bank_on_disk),
        "e293_bank_matches_frozen": bool(bank_on_disk == E293_BANK),
        "fresh_bank_members_in_e293_bank":
            {nm: bool(nm in bank_on_disk) for nm in fresh_bank},
        "counts": counts,
        "counts_match_frozen": bool(all(
            counts[nm] == exp for nm, _, _, exp in PANEL)),
        "chars_in_vocab": {nm: all(c in stoi for c in nm)
                           for nm, *_ in PANEL},
        "seven_letters": {nm: len(nm) == 7 for nm in synth_all},
        "not_g1_name": {nm: nm != G1.NAME for nm in synth_all},
        "initials": initials,
        "initials_distinct": bool(len(set(initials)) == len(initials)),
        "scrambled_is_permutation":
            bool(sorted(scrambled) == sorted("TAVIREN")),
        "scrambled_not_a_bank_name": bool(scrambled not in E293_BANK),
    }
    gpanel["pass"] = bool(
        bank_on_disk == E293_BANK
        and all(nm in bank_on_disk for nm in fresh_bank)
        and gpanel["counts_match_frozen"]
        and all(gpanel["chars_in_vocab"].values())
        and all(gpanel["seven_letters"].values())
        and all(gpanel["not_g1_name"].values())
        and gpanel["initials_distinct"]
        and gpanel["scrambled_is_permutation"]
        and gpanel["scrambled_not_a_bank_name"])
    METRICS["gates"]["G_PANEL"] = gpanel
    if not gpanel["pass"]:
        raise texture(f"G_PANEL FAILURE: {gpanel}")
    gnamefree = {"synthetic_counts": {nm: counts[nm] for nm in synth_all},
                 "pass": bool(all(counts[nm] == 0 for nm in synth_all))}
    METRICS["gates"]["G_NAMEFREE"] = gnamefree
    if not gnamefree["pass"]:
        raise texture(f"G_NAMEFREE FAILURE: {gnamefree}")
    log(f"P0: G_PANEL + G_NAMEFREE PASS — 11 names, initials {''.join(initials)}"
        f" all distinct; bank on disk == frozen {E293_BANK}; corpus counts "
        f"exact (KING {counts['KING']}, LEONTES {counts['LEONTES']}, "
        f"MAMILLIUS {counts['MAMILLIUS']}, ELIZABETH {counts['ELIZABETH']}, "
        f"FLORIZEL {counts['FLORIZEL']})")

    # the host-g0 site (e261's splice bank verbatim — x23's construction)
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
    host_ids = torch.stack(
        [corpus.encode(train_text[p - G1.PRE: p]) for p, _ in install_occ])
    gbatt = {
        "form": "THE HOST-G0 SITE: e261's splice bank verbatim (the exact "
                "bank on which KK/KT were committed)",
        "install_mix": mix,
        "shape": list(host_ids.shape),
        "expected": {"mix": {"FLORIZEL": 19, "ELIZABETH": 41},
                     "shape": [60, 130]},
        "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                     and list(host_ids.shape) == [60, 130]),
    }
    METRICS["gates"]["G_BATTERY"] = gbatt
    if not gbatt["pass"]:
        raise texture(f"G_BATTERY FAILURE: {gbatt}")

    # the neutral site (frozen rule, seed 24001)
    look = 30
    cands = [p for p in range(G1.PRE, len(train_text))
             if train_text[p].isupper() and train_text[p].isascii()
             and train_text[p].isalpha()
             and not any(h in train_text[p - G1.PRE - 8: p + look]
                         for h in G1.HOSTS)]
    nrng = random.Random(NEUTRAL_SEED)
    nrng.shuffle(cands)
    neutral_pos = cands[:60]
    neutral_ids = torch.stack(
        [corpus.encode(train_text[p - G1.PRE: p]) for p in neutral_pos])
    upper_ok = sum(1 for p in neutral_pos if train_text[p].isupper())
    hostfree_ok = sum(
        1 for p in neutral_pos
        if not any(h in train_text[p - G1.PRE - 8: p + look]
                   for h in G1.HOSTS))
    gneut = {
        "form": "THE NEUTRAL SITE: 60 train-split windows, 130 chars, "
                "ending at an uppercase ASCII letter, host-free in window "
                "+ 30-char lookahead; random.Random(24001) shuffle, "
                "first 60",
        "n_candidates": len(cands),
        "shape": list(neutral_ids.shape),
        "next_char_upper": f"{upper_ok}/60",
        "host_free": f"{hostfree_ok}/60",
        "pass": bool(list(neutral_ids.shape) == [60, 130]
                     and upper_ok == 60 and hostfree_ok == 60
                     and len(cands) >= 60),
    }
    METRICS["gates"]["G_NEUTRAL"] = gneut
    if not gneut["pass"]:
        raise texture(f"G_NEUTRAL FAILURE: {gneut}")
    sites = {"host_g0": host_ids, "neutral": neutral_ids}
    log(f"P0: G_BATTERY + G_NEUTRAL PASS — host mix 19/41 shape [60,130]; "
        f"neutral {len(cands)} candidates -> 60, all upper-next, host-free")
    write_partial("P0 panel + both sites gated")

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)

    # ============ P0b: the parents hard-bound (Rule 12) =================
    x15m = json.loads(X15_METRICS.read_text(encoding="utf-8"))
    x23m = json.loads(X23_METRICS.read_text(encoding="utf-8"))
    x17m = json.loads(X17_METRICS.read_text(encoding="utf-8"))
    e311m = json.loads(E311_METRICS.read_text(encoding="utf-8"))
    e311_fresh = e311m["gates"]["G_FRESH"]

    def near(a, b, tol):
        return abs(float(a) - float(b)) <= tol

    lit_checks = {
        "x15_diag_kk": (x15m["verdict"]["full_dose_complement_t0_g0"],
                        X15_DIAG_KK, 1e-18),
        "x15_s_full": (x15m["dose_match"]["s_full"], X15_S_FULL, 1e-15),
        "x15_comp_l2": (x15m["complement_regenerated"]["l2_fp64"],
                        X15_COMP_L2_64, 1e-12),
        "x15_fullread": (x15m["gates"]["G_FULLREAD"]["committed"],
                         K10K_FULL_G0_Z, 1e-15),
        "x15_base_g0_z": (x15m["anchors"]["base_e001_t0"]["g0"]["mean_pz"],
                          K10K_BASE_G0_Z, 1e-18),
        "x15_verdict": (x15m["verdict"]["word"], "DEAD-AT-FULL-DOSE", None),
        "x23_matrix_kk": (x23m["matrix"]["KK"], X15_DIAG_KK, 1e-18),
        "x23_matrix_kt": (x23m["matrix"]["KT"], X23_KT, 1e-18),
        "x23_base_on_t": (x23m["calibrations"]["base_on_T"]["mine"],
                          X17_BASE_G0_T, 1e-18),
        "x23_verdict": (x23m["verdict"]["word"], "EFFECT-FOLLOWS-WRITE",
                        None),
        "e311_fresh_g0_t": (e311_fresh["read_g0_t"], E311_FRESH_G0_T,
                            1e-15),
        "x17_tavfull_mine": (x17m["gates"]["G_TAVFULL"]["mine_g0_pT"],
                             E311_FRESH_G0_T, FULLREAD_TOL),
    }

    def lit_ok(v) -> bool:
        return v[0] == v[1] if v[2] is None else near(v[0], v[1], v[2])

    binds = [
        ("e001", CKPT_DIR / BASE_CK, BASE_MD5),
        ("e261_K10K_inst_resume", CKPT_DIR / FACT_CK, FACT_MD5),
        ("e264_rooms", CKPT_DIR / ROOMS264_CK, ROOMS264_MD5),
        ("x15_metrics", X15_METRICS, X15_METRICS_MD5),
        ("x15_comp_xFULL_carrier", X15_COMP_CARRIER, X15_COMP_CARRIER_MD5),
        ("x23_metrics", X23_METRICS, X23_METRICS_MD5),
        ("x17_metrics", X17_METRICS, X17_METRICS_MD5),
        ("x17_fullwrite_carrier", X17_FULL_CARRIER, X17_FULL_CARRIER_MD5),
        ("e311_metrics", E311_METRICS, E311_METRICS_MD5),
        ("e293_bank_source", E293_SOURCE, E293_SOURCE_MD5),
    ]
    bind_records = {}
    for nm, p, b in binds:
        got = md5of(p)
        bind_records[nm] = {"path": str(p.relative_to(REPO)), "md5": got,
                            "bound": b, "match": got == b}
        if got != b:
            raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} {got} != {b}")
    gendp = {
        "form": "every parent artifact md5-bound (the displacement's own "
                "carrier + the fullT control carrier + the e293 bank "
                "source included) + every committed reference READ FROM "
                "THE ARTIFACT and cross-checked against the frozen "
                "literals",
        "md5_binds": bind_records,
        "literal_crosscheck": {
            k: {"artifact": v[0], "frozen": v[1], "tol": v[2],
                "match": lit_ok(v)}
            for k, v in lit_checks.items()},
        "pass": bool(all(r["match"] for r in bind_records.values())
                     and all(lit_ok(v) for v in lit_checks.values())),
    }
    if not gendp["pass"]:
        raise texture(f"G_ENDPOINTS failure: {gendp}")
    METRICS["gates"]["G_ENDPOINTS"] = gendp
    log(f"P0b: G_ENDPOINTS PASS — {len(binds)} parents md5-bound (both "
        f"carriers + the bank source); {len(lit_checks)} committed "
        f"references cross-checked from artifacts, never retyped")
    write_partial("P0b parents hard-bound")

    # ============ P1: the base organism + the panel priors ==============
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    N = base_net.num_params()
    named_p = list(base_net.named_parameters())
    sd_keys = list(base_sd.keys())
    gflat = {
        "params_count": len(named_p),
        "key_order_matches_parameters": bool([k for k, _ in named_p]
                                             == sd_keys),
        "n_params": N,
        "pass": bool([k for k, _ in named_p] == sd_keys
                     and len(sd_keys) == len(named_p)
                     and N == N_PARAMS),
    }
    if not gflat["pass"] or N != N_PARAMS:
        raise texture(f"FLAT-BASIS GATE FAILURE: {gflat}")
    METRICS["gates"]["G_FLATBASIS"] = gflat

    base_evl = G1.evl_load(base_sd)
    site_passes = {"base": {s: site_read(base_evl, ids)
                            for s, ids in sites.items()}}
    ce_base = float(G1.ce_fixed_cpu(base_evl, r_eval_x, r_eval_y))
    del base_evl
    base_reads = {nm: {s: name_read(site_passes["base"][s], stoi[nm[0]])
                       for s in sites} for nm, *_ in PANEL}

    # G_BASEPRIOR: the two committed priors + e311's 0.05 bar on synthetics
    bz = base_reads["ZEPHYRA"]["host_g0"]["mean_pz"]
    bt = base_reads["TAVIREN"]["host_g0"]["mean_pz"]
    gbase = {
        "form": "the base organism's own priors (e311's G_BASE convention, "
                "banded <= 0.05 on every synthetic name at host-g0) + the "
                "two committed priors cross-checked in [0.5x, 2x]",
        "anchor_prior_host": {"mine": bz, "committed": K10K_BASE_G0_Z,
                              "ratio": bz / K10K_BASE_G0_Z,
                              "band": list(REPRO_BAND)},
        "taviren_prior_host": {"mine": bt, "committed": X17_BASE_G0_T,
                               "ratio": bt / X17_BASE_G0_T,
                               "band": list(REPRO_BAND)},
        "max_synthetic_prior_host": max(
            base_reads[nm]["host_g0"]["mean_pz"] for nm in synth_all),
        "bar": BASE_BAR,
        "pass": bool(REPRO_BAND[0] <= bz / K10K_BASE_G0_Z
                     <= REPRO_BAND[1]
                     and REPRO_BAND[0] <= bt / X17_BASE_G0_T
                     <= REPRO_BAND[1]
                     and all(base_reads[nm]["host_g0"]["mean_pz"] <= BASE_BAR
                             for nm in synth_all)),
    }
    if not gbase["pass"]:
        raise texture(f"G_BASEPRIOR FAILURE: {gbase}")
    METRICS["gates"]["G_BASEPRIOR"] = gbase

    # G_PANELPRIOR: e293's bank-discipline band (co-report; straddles are
    # disclosed deviations, NOT halts — no spare bank members)
    band_report = {nm: base_reads[nm]["host_g0"]["mean_pz"]
                   for nm in synth_all}
    straddles = [nm for nm, v in band_report.items()
                 if BANK_BAND < v <= BASE_BAR]
    gpanelp = {
        "form": "e293's bank-discipline band: fresh synthetics + scrambled "
                "+ TAVIREN base prior at host-g0 <= 0.004 (co-report; a "
                "straddle in (0.004, 0.05] is a disclosed deviation — no "
                "spare bank members)",
        "priors_host_g0": band_report,
        "band": BANK_BAND,
        "within_band": {nm: bool(v <= BANK_BAND)
                        for nm, v in band_report.items()},
        "straddles": straddles,
        "pass": True,   # informational gate: the 0.05 halt is G_BASEPRIOR
    }
    METRICS["gates"]["G_PANELPRIOR"] = gpanelp
    band_str = ", ".join(f"{nm}={v:.2e}" for nm, v in band_report.items())
    log(f"P1: G_FLATBASIS + G_BASEPRIOR + G_PANELPRIOR PASS (N={N}; prior "
        f"Z {bz:.4e} vs {K10K_BASE_G0_Z:.4e}; prior T {bt:.6f} vs "
        f"{X17_BASE_G0_T:.6f}; synthetic host priors {band_str}; straddles "
        f"{straddles})")
    write_partial("P1 base + panel priors gated")

    # ============ P2: THE ROOM ===========================================
    room = SRCT(N_PARAMS, ROOM_K, ROOM_SEED_D, ROOM_SEED_S)
    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    D264 = _to_np(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = _to_np(rooms264["model"]["K10K"]["S"])
    del rooms264
    cert_rng = np.random.default_rng(CERT_SEED)
    idem, kept2 = [], []
    for _ in range(2):
        x = cert_rng.standard_normal(N_PARAMS)
        px = room.project(x)
        ppx = room.project(px)
        idem.append(float(np.linalg.norm(ppx - px) / np.linalg.norm(px)))
        kept2.append(float((px @ px) / (x @ x)))
    groom = {
        "form": "the committed K10K room (SRCT k=10,000, seeds 26113/26114): "
                "D/S bit-identical to e264_rooms.pt; light in-cell "
                "certification (x17/x23's convention)",
        "D_bit_equal": bool(np.array_equal(room.D, D264)),
        "S_bit_equal": bool(np.array_equal(room.S, S264)),
        "idempotency_max": max(idem),
        "kept2_mean": float(np.mean(kept2)),
        "kept2_expect": ROOM_K / N_PARAMS,
        "pass": bool(np.array_equal(room.D, D264)
                     and np.array_equal(room.S, S264)
                     and max(idem) <= 1e-8
                     and abs(float(np.mean(kept2)) - ROOM_K / N_PARAMS)
                     <= 5.0 * math.sqrt(2.0 * ROOM_K) / N_PARAMS),
    }
    if not groom["pass"]:
        raise texture(f"ROOM GATE FAILURE: {groom}")
    METRICS["gates"]["G_ROOM"] = groom
    log(f"P2: G_ROOM PASS — K10K room D/S BIT-BOUND (idem {max(idem):.1e}; "
        f"kept2 {np.mean(kept2):.6f} vs {ROOM_K / N_PARAMS:.6f})")

    # ============ P3: THE K10K WRITE + ITS COMPLEMENT (x15 verbatim) =====
    fact_art = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                          weights_only=False)
    fact_sd = {k: v.detach().clone() for k, v in fact_art["model"].items()}
    del fact_art
    dW_k64 = {k: fact_sd[k].double() - base_sd[k].double() for k in sd_keys}
    dW_k_flat = torch.cat([dW_k64[k].reshape(-1)
                           for k in sd_keys]).numpy()
    DWK_L2 = float(np.linalg.norm(dW_k_flat))
    if abs(DWK_L2 - K10K_WRITE_NORM) > 1e-9:
        raise texture(f"K10K write norm {DWK_L2} != {K10K_WRITE_NORM}")
    E_FULL_K = float(dW_k_flat @ dW_k_flat)
    pin_k = room.project(dW_k_flat)
    comp_k_flat = dW_k_flat - pin_k
    E_IN_K = float(pin_k @ pin_k)
    E_OUT_K = float(comp_k_flat @ comp_k_flat)
    gorth = {
        "energy_sum_rel_resid": abs(E_IN_K + E_OUT_K - E_FULL_K) / E_FULL_K,
        "cross_dot_rel": abs(float(pin_k @ comp_k_flat)) / E_FULL_K,
        "bar": 1e-9,
    }
    gorth["pass"] = bool(gorth["energy_sum_rel_resid"] < 1e-9
                         and gorth["cross_dot_rel"] < 1e-9)
    if not gorth["pass"]:
        raise texture(f"G_ORTH (K10K write) FAILURE: {gorth}")
    METRICS["gates"]["G_ORTH"] = gorth

    COMP_K_L2 = float(np.linalg.norm(comp_k_flat))
    s_full_k = DWK_L2 / COMP_K_L2                    # x15's exact arithmetic
    scaled_k64 = s_full_k * comp_k_flat

    def unflat_like_base(flat64: np.ndarray) -> dict:
        out, off = {}, 0
        for k in sd_keys:
            n = base_sd[k].numel()
            out[k] = torch.from_numpy(
                np.ascontiguousarray(flat64[off:off + n])) \
                .reshape(base_sd[k].shape)
            off += n
        return out

    scaled_k32 = {k: v.float()
                  for k, v in unflat_like_base(scaled_k64).items()}
    fl_k32 = np.concatenate([scaled_k32[k].double().numpy().reshape(-1)
                             for k in sd_keys])
    gdose = {
        "form": "||s_full_k * comp_k|| == ||dW_k|| to 1e-12 (fp64, derived "
                "never trusted)",
        "s_full": s_full_k, "committed": X15_S_FULL,
        "s_full_abs_diff": abs(s_full_k - X15_S_FULL),
        "scaled_fp64_l2": float(np.linalg.norm(scaled_k64)),
        "write_l2": DWK_L2,
        "abs_diff": abs(float(np.linalg.norm(scaled_k64)) - DWK_L2),
        "bar": 1e-12,
        "pass": bool(abs(float(np.linalg.norm(scaled_k64)) - DWK_L2) <= 1e-12
                     and abs(s_full_k - X15_S_FULL) <= 1e-15),
    }
    ginroom = {
        "form": "x15's exact G_INROOM_SCALED form (in-room ENERGY share "
                "<= 1e-12; fp64 intended AND fp32 injected)",
        "scaled_fp64": inroom_energy_share(room, scaled_k64),
        "fp32_injected": inroom_energy_share(room, fl_k32),
        "co_norm_fracs": {"fp64": inroom_frac(room, scaled_k64),
                          "fp32": inroom_frac(room, fl_k32)},
        "bar": 1e-12,
    }
    ginroom["pass"] = bool(ginroom["scaled_fp64"] <= 1e-12
                           and ginroom["fp32_injected"] <= 1e-12)
    if not (gdose["pass"] and ginroom["pass"]):
        raise texture(f"DOSE/INROOM FAILURE: {gdose} {ginroom}")
    METRICS["gates"]["G_DOSEMATCH"] = gdose
    METRICS["gates"]["G_INROOM"] = ginroom
    log(f"P3: K10K complement rebuilt x15-verbatim (s_full {s_full_k:.10f} "
        f"vs committed {X15_S_FULL:.10f}; in-room energy "
        f"{ginroom['scaled_fp64']:.1e})")

    # ============ P4: G_REGEN — the complement vs its CARRIER ==========
    ck_k = torch.load(X15_COMP_CARRIER, map_location="cpu",
                      weights_only=False)
    ck_fw = torch.load(X17_FULL_CARRIER, map_location="cpu",
                       weights_only=False)
    bit_equal = {k: bool(torch.equal(ck_k["delta"][k].float(), scaled_k32[k]))
                 for k in sd_keys}
    model_ok = all(torch.equal(ck_k["model"][k],
                               base_sd[k] + ck_k["delta"][k])
                   for k in sd_keys)
    gregen = {
        "form": "the regenerated full-dose complement (fp32) vs the "
                "committed carrier's on-disk delta — bit-equal on EVERY "
                "key; + natural fp64 comp L2 vs committed (1e-9); + "
                "carrier model == base + delta re-derived",
        "carrier": str(X15_COMP_CARRIER.relative_to(REPO)),
        "keys_bit_equal": int(sum(bit_equal.values())),
        "keys_total": len(sd_keys),
        "all_keys_bit_equal": all(bit_equal.values()),
        "my_comp64_l2": COMP_K_L2,
        "committed_l2": X15_COMP_L2_64,
        "l2_abs_diff": abs(COMP_K_L2 - X15_COMP_L2_64),
        "l2_bar": 1e-9,
        "carrier_model_eq_base_plus_delta": bool(model_ok),
        "pass": bool(all(bit_equal.values())
                     and abs(COMP_K_L2 - X15_COMP_L2_64) < 1e-9 and model_ok),
    }
    if not gregen["pass"]:
        raise texture(f"G_REGEN FAILURE: {gregen}")
    METRICS["gates"]["G_REGEN"] = gregen
    log("P4: G_REGEN PASS — the displacement BIT-EQUAL to x15's committed "
        "carrier (every key); carrier model == base + delta re-derived")
    write_partial("P4 displacement bit-bound to its carrier")

    # ============ P5: THE STATES — one applied state, many probes =======
    states = {}
    states["base"] = {
        "site_passes": site_passes["base"], "ce_r": ce_base,
        "source": "in-memory e001 (md5-bound)",
    }
    comp_net = G1.evl_load(ck_k["model"])   # THE displacement — ONE net
    states["comp"] = {
        "site_passes": {s: site_read(comp_net, ids)
                        for s, ids in sites.items()},
        "ce_r": float(G1.ce_fixed_cpu(comp_net, r_eval_x, r_eval_y)),
        "source": "runs/x15/x15_comp_xFULL.pt ['model'] (md5-bound; "
                  "bit-regenerated + verified) — THE ONE applied "
                  "displacement state, every name read probes THIS net",
    }
    del comp_net
    fullT_net = G1.evl_load(ck_fw["model"])
    states["fullT"] = {
        "site_passes": {s: site_read(fullT_net, ids)
                        for s, ids in sites.items()},
        "ce_r": float(G1.ce_fixed_cpu(fullT_net, r_eval_x, r_eval_y)),
        "source": "runs/x17/x17_x15R_fullwrite_control.pt ['model'] "
                  "(md5-bound)",
    }
    del fullT_net
    fullK_net = G1.evl_load(base_sd)                # x15's exact probe path
    with torch.no_grad():
        for n, p in fullK_net.named_parameters():
            p.add_(dW_k64[n].float())
    fact_bitexact = all(torch.equal(fullK_net.state_dict()[k], fact_sd[k])
                        for k in sd_keys)
    states["fullK"] = {
        "site_passes": {s: site_read(fullK_net, ids)
                        for s, ids in sites.items()},
        "ce_r": float(G1.ce_fixed_cpu(fullK_net, r_eval_x, r_eval_y)),
        "source": "in-memory base + dW_k (x15's G_FULLREAD path; bit-equal "
                  f"to the e261 model: {fact_bitexact})",
    }
    del fullK_net
    gone_state = {
        "form": "the displacement is ONE applied state (the committed "
                "carrier's own model dict, loaded once) probed at both "
                "sites on all 11 names — no per-name reconstruction; "
                "identical by construction, provenance recorded",
        "pass": True,
    }
    METRICS["gates"]["G_ONESTATE"] = gone_state
    log("P5: 4 states read (base / comp [THE displacement, one net] / "
        "fullK / fullT); ce_r — "
        + " ".join(f"{k} {v['ce_r']:.3f}" for k, v in states.items()))

    # ============ P6: the census table ===================================
    reads = {}
    for nm, cls, cohort, _ in PANEL:
        cid = stoi[nm[0]]
        reads[nm] = {"class": cls, "cohort": cohort, "cid": cid}
        for st in states:
            for s in sites:
                reads[nm][f"{st}_{s}"] = name_read(
                    states[st]["site_passes"][s], cid)
    for nm in reads:
        for s in sites:
            pb = reads[nm][f"base_{s}"]["mean_pz"]
            pc = reads[nm][f"comp_{s}"]["mean_pz"]
            pf_k = reads[nm][f"fullK_{s}"]["mean_pz"]
            pf_t = reads[nm][f"fullT_{s}"]["mean_pz"]
            reads[nm][f"lift_{s}"] = pc / max(pb, PRIOR_FLOOR)
            reads[nm][f"gain_{s}"] = pc - pb
            reads[nm][f"dlogit_{s}"] = _logit(pc) - _logit(pb)
            reads[nm][f"lift_fullK_{s}"] = pf_k / max(pb, PRIOR_FLOOR)
            reads[nm][f"lift_fullT_{s}"] = pf_t / max(pb, PRIOR_FLOOR)
            reads[nm][f"prior_floored_{s}"] = bool(pb < PRIOR_FLOOR)
    METRICS["panel_reads"] = reads
    write_partial("P6 the census read (4 states x 2 sites x 11 names)")

    # G_FULLREAD: the two committed full-write controls
    fz = reads["ZEPHYRA"]["fullK_host_g0"]["mean_pz"]
    ft = reads["TAVIREN"]["fullT_host_g0"]["mean_pz"]
    gfull = {
        "form": "the two committed full-write controls on their own names "
                "(x15's own G_FULLREAD bar |d| <= 5e-3) — the instrument "
                "certified before any clause counts",
        "fullK_on_ZEPHYRA_host": {"mine": fz, "committed": K10K_FULL_G0_Z,
                                  "abs_diff": abs(fz - K10K_FULL_G0_Z),
                                  "bar": FULLREAD_TOL,
                                  "pass": bool(abs(fz - K10K_FULL_G0_Z)
                                               <= FULLREAD_TOL)},
        "fullT_on_TAVIREN_host": {"mine": ft, "committed": E311_FRESH_G0_T,
                                  "abs_diff": abs(ft - E311_FRESH_G0_T),
                                  "bar": FULLREAD_TOL,
                                  "pass": bool(abs(ft - E311_FRESH_G0_T)
                                               <= FULLREAD_TOL)},
    }
    gfull["pass"] = bool(gfull["fullK_on_ZEPHYRA_host"]["pass"]
                         and gfull["fullT_on_TAVIREN_host"]["pass"])
    METRICS["gates"]["G_FULLREAD"] = gfull
    if not gfull["pass"]:
        raise texture(f"G_FULLREAD FAILURE: {gfull}")

    # G_DIAG: the two committed complement references reproduced
    kk = reads["ZEPHYRA"]["comp_host_g0"]["mean_pz"]
    kt = reads["TAVIREN"]["comp_host_g0"]["mean_pz"]
    gdiag = {
        "form": "the displacement's two committed references reproduce "
                "in-session (x15's G_COMPREF band [0.5x, 2x])",
        "KK_ZEPHYRA": {"mine": kk, "committed": X15_DIAG_KK,
                       "ratio": kk / X15_DIAG_KK, "band": list(REPRO_BAND),
                       "pass": bool(REPRO_BAND[0] * X15_DIAG_KK
                                    <= kk <= REPRO_BAND[1] * X15_DIAG_KK)},
        "KT_TAVIREN": {"mine": kt, "committed": X23_KT,
                       "ratio": kt / X23_KT, "band": list(REPRO_BAND),
                       "pass": bool(REPRO_BAND[0] * X23_KT
                                    <= kt <= REPRO_BAND[1] * X23_KT)},
    }
    gdiag["pass"] = bool(gdiag["KK_ZEPHYRA"]["pass"]
                         and gdiag["KT_TAVIREN"]["pass"])
    METRICS["gates"]["G_DIAG"] = gdiag
    if not gdiag["pass"]:
        raise texture(f"G_DIAG FAILURE: {gdiag}")
    log(f"P6b: G_FULLREAD + G_DIAG PASS — fullK|Z {fz:.6f} (|d| "
        f"{abs(fz - K10K_FULL_G0_Z):.1e}); fullT|T {ft:.6f} (|d| "
        f"{abs(ft - E311_FRESH_G0_T):.1e}); comp|Z {kk:.6g} (x"
        f"{kk / X15_DIAG_KK:.4f}); comp|T {kt:.6g} (x{kt / X23_KT:.4f})")
    write_partial("P6b full-write + committed-reference gates PASSED")

    if SMOKE:
        METRICS["status"] = ("SMOKED — full gate path + all reads "
                             "exercised; NOTHING adjudicated")
        write_partial("SMOKE COMPLETE — nothing adjudicated")
        log("SMOKE COMPLETE (nothing adjudicated)")
        raise SystemExit(0)

    # ============ P7: ADJUDICATION (frozen) =============================
    def lifts_at(s):
        return {nm: reads[nm][f"lift_{s}"] for nm, *_ in PANEL}

    clause_eval = {}
    for s in sites:
        la = lifts_at(s)
        cohort_a = [la[nm] for nm in COHORT_A]
        anchor_lift = la[ANCHOR]
        med = float(np.median(cohort_a))
        lo = min([anchor_lift] + cohort_a)
        hi = max([anchor_lift] + cohort_a)
        c_c = {nm: bool(lo <= la[nm] <= hi) for nm in COHORT_C}
        clause_eval[s] = {
            "anchor_lift": anchor_lift,
            "cohort_a_lifts": {nm: la[nm] for nm in COHORT_A},
            "cohort_a_spread": max(cohort_a) / min(cohort_a),
            "cohort_a_median": med,
            "clause_A_spread_le_3": bool(max(cohort_a) / min(cohort_a)
                                         <= SPREAD_BAR),
            "clause_B_anchor_le_0p2x_median":
                bool(anchor_lift <= ANCHOR_FRAC * med),
            "interval": [lo, hi],
            "clause_C_corpus_inside": c_c,
            "clause_C_pass": bool(all(c_c.values())),
            "priors": {nm: reads[nm][f"base_{s}"]["mean_pz"]
                       for nm, *_ in PANEL},
        }
    ogc_site = {s: bool(clause_eval[s]["clause_A_spread_le_3"]
                        and clause_eval[s]["clause_B_anchor_le_0p2x_median"]
                        and clause_eval[s]["clause_C_pass"])
                for s in sites}
    one_generic = bool(all(ogc_site.values()))

    # structure bar: same-prior (within 3x both ways) pairs with >10x lift
    struct_pairs = []
    for s in sites:
        pr = {nm: reads[nm][f"base_{s}"]["mean_pz"] for nm, *_ in PANEL}
        li = lifts_at(s)
        for i, (ni, *_) in enumerate(PANEL):
            for nj, *_ in PANEL[i + 1:]:
                pi, pj = pr[ni], pr[nj]
                if (pi <= SAME_PRIOR_X * pj and pj <= SAME_PRIOR_X * pi):
                    lr = max(li[ni], li[nj]) / min(li[ni], li[nj])
                    if lr > STRUCT_SPREAD:
                        struct_pairs.append(
                            {"site": s, "pair": [ni, nj],
                             "priors": [pi, pj], "lifts": [li[ni], li[nj]],
                             "lift_ratio": lr})
    structure = bool(len(struct_pairs) > 0)

    if one_generic:
        word = "ONE-GENERIC-CURVE"
        clause = (
            "the lift ratios of all count-0/low-prior names fall within "
            f"~3x of each other (host spread "
            f"{clause_eval['host_g0']['cohort_a_spread']:.2f}x, neutral "
            f"{clause_eval['neutral']['cohort_a_spread']:.2f}x), the "
            f"formed anchor's lift is <= 0.2x theirs, and the mid/high-"
            "prior corpus names' lifts fall between — prior-fragility is "
            "generic; NECESSITY RESTORES with the rider 'reads are scored "
            "prior-relative (a complement floors below ~10x prior; a full "
            "write clears it)'; tonight's write-dependence scare re-closes "
            "as an instrument lesson")
    elif structure:
        word = "NAME-SPECIFIC-STRUCTURE"
        worst = max(struct_pairs, key=lambda d: d["lift_ratio"])
        clause = (
            f"some marginal names lift >> others — {worst['pair'][0]} vs "
            f"{worst['pair'][1]} at {worst['site']} (priors "
            f"{worst['priors'][0]:.3g}/{worst['priors'][1]:.3g}, lifts "
            f"{worst['lifts'][0]:.1f}x/{worst['lifts'][1]:.1f}x, "
            f"{worst['lift_ratio']:.1f}x spread among same-prior names) — "
            "fragility is structure, not prior; the census continues with "
            "name-geometry probes; necessity stays weakened")
    else:
        word = "GAP"
        clause = (
            "neither frozen bar fires — report verbatim; no claim wording "
            "change without a new registered cell")
    p_score = {
        "P_x24a_statement": REGISTERED["P_x24a_verbatim"],
        "P_x24a_outcome": ("HIT — ONE-GENERIC-CURVE"
                           if word == "ONE-GENERIC-CURVE" else
                           f"MISS — the verdict is {word}"),
        "P_x24a_exec_statement": REGISTERED["executor_position"],
        "P_x24a_exec_outcome": "",   # filled below
    }

    # riders: monotone curve + anchor formed-fraction + the 10x-prior census
    riders = {}
    for s in sites:
        pr = [math.log10(max(reads[nm][f"base_{s}"]["mean_pz"], 1e-30))
              for nm, *_ in PANEL]
        lr = [math.log10(reads[nm][f"lift_{s}"]) for nm, *_ in PANEL]
        riders[f"spearman_logprior_loglift_{s}"] = spearman(pr, lr)
    anchor_ff = {s: reads["ZEPHYRA"][f"comp_{s}"]["mean_pz"]
                 / reads["ZEPHYRA"][f"fullK_{s}"]["mean_pz"] for s in sites}
    rider10 = {s: {nm: bool(reads[nm][f"lift_{s}"] < 10.0
                            and reads[nm][f"lift_fullK_{s}"] > 10.0)
                   for nm, *_ in PANEL} for s in sites}
    riders["anchor_formed_fraction"] = anchor_ff
    riders["x10prior_rider_census"] = rider10
    METRICS["riders"] = riders

    if word == "GAP":
        p_score["P_x24a_exec_outcome"] = (
            "HIT — executor predicted GAP (clause B strain registered "
            "pre-compute) with the monotone rider "
            f"(rhos {riders['spearman_logprior_loglift_host_g0']:+.2f}/"
            f"{riders['spearman_logprior_loglift_neutral']:+.2f})"
            if (not one_generic and not structure) else "")
    elif word == "ONE-GENERIC-CURVE":
        p_score["P_x24a_exec_outcome"] = (
            "MISS — the executor's clause-B strain did not materialize as "
            "a verdict-killer; the lab guess wins")
    else:
        p_score["P_x24a_exec_outcome"] = (
            "MISS — the executor predicted GAP; the verdict is "
            "NAME-SPECIFIC-STRUCTURE")

    METRICS["verdict"] = {
        "word": word,
        "clause": clause,
        "clause_eval": clause_eval,
        "one_generic_per_site": ogc_site,
        "structure_pairs": struct_pairs,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "clauses_fixed": REGISTERED["clauses_fixed"],
        "prediction": p_score,
    }
    write_partial(f"P7 ADJUDICATED: {word}")

    # ============ P8: the figure ========================================
    import matplotlib                                 # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                   # noqa: E402

    cls_style = {
        "anchor_formed_host": ("k", "*", 230, "anchor (ZEPHYRA, the K10K "
                               "write's own name)"),
        "synthetic_count0": ("steelblue", "o", 90,
                             "count-0 synthetic (TAVIREN = committed ref)"),
        "synthetic_count0_fresh_bank": ("royalblue", "o", 90,
                                        "count-0 synthetic, FRESH bank"),
        "scrambled_control": ("purple", "s", 90, "scrambled control"),
        "corpus_rare_seen": ("seagreen", "^", 110, "corpus rare (13)"),
        "corpus_mid_proper": ("darkorange", "D", 80, "corpus mid (125)"),
        "corpus_common_token": ("gold", "D", 110, "corpus common (556)"),
        "corpus_host_high": ("crimson", "P", 130, "corpus host (105/45)"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), sharey=True)
    for ax, s, ttl in ((axes[0], "host_g0",
                        "(a) HOST-G0 site (the 60 install contexts — the "
                        "committed bank)"),
                       (axes[1], "neutral",
                        "(b) NEUTRAL site (60 host-free corpus capitals)")):
        la = lifts_at(s)
        for nm, cls, _, _ in PANEL:
            col, mk, sz, _ = cls_style[cls]
            x = reads[nm][f"base_{s}"]["mean_pz"]
            ax.scatter(x, la[nm], c=col, marker=mk, s=sz, zorder=3,
                       edgecolors="k", linewidths=0.5)
            if nm in ("ZEPHYRA", "TAVIREN"):
                ax.annotate(f" {nm}\n {la[nm]:.1f}x", (x, la[nm]),
                            fontsize=7.5, va="bottom", ha="left")
            # the full-write comparison point (open marker)
            ax.scatter(x, reads[nm][f"lift_fullK_{s}"], facecolors="none",
                       edgecolors=col, marker="o", s=45, alpha=0.55,
                       linewidths=1.1, zorder=2)
        ax.axhline(10.0, color="gray", ls=":", lw=1.2,
                   label="the x10-prior rider line (lift = 10)")
        ax.axhline(1.0, color="gray", ls="-", lw=0.5, alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("base prior p(name[0]) at the site (log)")
        ax.grid(alpha=0.25, which="both")
        ax.set_title(ttl + f"  — Spearman(log-prior, log-lift) = "
                     f"{riders[f'spearman_logprior_loglift_{s}']:+.2f}",
                     fontsize=9.5)
    axes[0].set_ylabel("lift  p_comp / p_base (log; filled)  &  "
                       "p_fullK / p_base (open)")
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="",
                          markersize=7 if m in ("o", "s") else 10,
                          markeredgecolor="k",
                          label=lbl)
               for c, m, _, lbl in cls_style.values()]
    handles.append(plt.Line2D([], [], color="gray", ls=":", label=
                              "lift = 10 (the rider)"))
    axes[1].legend(handles=handles, fontsize=6.6, loc="lower left",
                   framealpha=0.9)
    fig.suptitle("x24 — THE PRIOR-FRAGILITY CENSUS: one fixed full-dose "
                 "displacement (the K10K complement, x15's carrier) x 11 "
                 f"names x 2 sites — verdict {word}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RD / "x24_prior_census.png", dpi=140)
    log("FIGURE written")

    # ============ P9: REPORT.md =========================================
    def fmt(v):
        return f"{v:.4g}" if v >= 1e-4 else f"{v:.3e}"

    tbl = ["| site | name | class | count | base prior | comp read | "
           "LIFT | dlogit | gain | fullK lift | fullT lift | argmax |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("host_g0", "neutral"):
        for nm, cls, _, cnt in PANEL:
            r = reads[nm]
            tbl.append(
                f"| {s} | **{nm}** | {cls} | {cnt} | "
                f"{fmt(r[f'base_{s}']['mean_pz'])} | "
                f"{fmt(r[f'comp_{s}']['mean_pz'])} | "
                f"**{r[f'lift_{s}']:.2f}x** | "
                f"{r[f'dlogit_{s}']:+.2f} | "
                f"{fmt(r[f'gain_{s}'])} | "
                f"{r[f'lift_fullK_{s}']:.2f}x | "
                f"{r[f'lift_fullT_{s}']:.2f}x | "
                f"{r[f'comp_{s}']['frac_argmax_z']:.2f} |")
    clause_tbl = ["| site | cohort-A spread (bar <= 3x) | anchor lift | "
                  "cohort-A median | clause B (<= 0.2x med) | interval | "
                  "clause C | all three |",
                  "|---|---|---|---|---|---|---|---|"]
    for s in ("host_g0", "neutral"):
        ce = clause_eval[s]
        clause_tbl.append(
            f"| {s} | {ce['cohort_a_spread']:.2f}x "
            f"({'PASS' if ce['clause_A_spread_le_3'] else 'FAIL'}) | "
            f"{ce['anchor_lift']:.2f}x | {ce['cohort_a_median']:.2f}x | "
            f"{'PASS' if ce['clause_B_anchor_le_0p2x_median'] else 'FAIL'} "
            f"| [{ce['interval'][0]:.2f}x, {ce['interval'][1]:.2f}x] | "
            f"{'PASS' if ce['clause_C_pass'] else 'FAIL'} | "
            f"{'PASS' if ogc_site[s] else 'FAIL'} |")
    gpass = sum(1 for g in METRICS["gates"].values()
                if isinstance(g, dict) and g.get("pass"))
    gtot = len(METRICS["gates"])
    rider10_host = ", ".join(nm for nm, ok in rider10["host_g0"].items()
                             if ok) or "(none)"
    rider10_neutral = ", ".join(nm for nm, ok in rider10["neutral"].items()
                                if ok) or "(none)"
    report = f"""# x24 — THE PRIOR-FRAGILITY CENSUS ({word})

**The question (T285):** is a name's lift under a fixed full-dose generic
displacement a generic function of its prior (or anchoring), or
name-specific structure? **The displacement:** the K10K write's complement
at full dose (x15's construction, bit-equal to its committed carrier,
probed from the carrier's own model dict — ONE applied state for every
name read). Its committed effect on two names: ZEPHYRA 0.0007308
("unmoved" in absolute terms), TAVIREN 0.0272878 (lifted) — both
reproduced in-session before any clause counted.

**The panel:** 11 names, 11 distinct initial slots — the anchor ZEPHYRA
(the displacement's parent write's own target name; a disclosed confound),
TAVIREN (committed reference), 3 FRESH e293-bank names (QELVARO / BUVONDI
/ NYSTORA — count-0, no relation to the write), the scrambled control
VIRETAN (TAVIREN's letters, initial V), and 5 corpus names with stated
train-split counts (MAMILLIUS 13, LEONTES 125, KING 556, ELIZABETH 105,
FLORIZEL 45). **Two reading sites:** the host's 60 g0 install contexts
(the committed bank) + a fresh NEUTRAL bank (60 host-free corpus
capitals, seed 24001).

## THE LIFT-vs-PRIOR CENSUS (read = p(name[0]), mean over 60 contexts)

{chr(10).join(tbl)}

## THE CLAUSES (frozen at birth)

{chr(10).join(clause_tbl)}

## Verdict: {word}

{clause}.

**P-x24a scoring (registered pre-compute, never shopped):**
{p_score['P_x24a_statement']} -> **{p_score['P_x24a_outcome']}**.
Executor: {p_score['P_x24a_exec_outcome']}.

**Riders (never bars):** Spearman(log-prior, log-lift) — host
{riders['spearman_logprior_loglift_host_g0']:+.3f}, neutral
{riders['spearman_logprior_loglift_neutral']:+.3f}; the anchor's
formed-fraction (p_comp/p_fullK on ZEPHYRA) — host
{anchor_ff['host_g0']:.4f}, neutral {anchor_ff['neutral']:.4f}; the
x10-prior rider census (comp < 10x AND fullK > 10x) — host:
{rider10_host}; neutral: {rider10_neutral}. ce_r — base
{states['base']['ce_r']:.3f}, comp
{states['comp']['ce_r']:.3f}, fullK {states['fullK']['ce_r']:.3f}, fullT
{states['fullT']['ce_r']:.3f}; top-1 p on host — base
{states['base']['site_passes']['host_g0']['rider_mean_top1_p']:.3f}, comp
{states['comp']['site_passes']['host_g0']['rider_mean_top1_p']:.3f}. No
organism collapsed; every read is a name-level fact.

## Gates: {gpass}/{gtot} PASS

G_ENDPOINTS ({len(binds)} md5 binds incl. the displacement carrier + the
fullT carrier + the e293 bank source; {len(lit_checks)} committed
references read from artifacts, never retyped), G_FLATBASIS, G_PANEL
(+G_NAMEFREE; e311's convention ported; initials distinct; counts exact),
G_BATTERY (host-g0 splice bank verbatim, 19/41, [60,130]), G_NEUTRAL
(seed 24001; all upper-next; host-free), G_BASEPRIOR (both committed
priors in [0.5x, 2x]; every synthetic <= 0.05), G_PANELPRIOR (e293's
0.004 band, co-report), G_ROOM (D/S bit-bound), G_ORTH, G_REGEN (the
displacement bit-equal to x15's carrier, every key), G_DOSEMATCH (1e-12),
G_INROOM (energy share <= 1e-12), G_FULLREAD (fullK|Z |d|
{abs(fz - K10K_FULL_G0_Z):.1e}; fullT|T |d| {abs(ft - E311_FRESH_G0_T):.1e}),
G_DIAG (comp|Z x{kk / X15_DIAG_KK:.4f}; comp|T x{kt / X23_KT:.4f}),
G_ONESTATE.

Disclosures: the read currency is the name-INITIAL char (the committed
KK/KT currency); lift = ratio for every clause (the dispatch's own
measure) — the committed values (anchor 54.6x vs TAVIREN 11.8x at host)
strained clause B pre-compute and that strain was registered at birth;
the anchor is the displacement's parent write's own name (confound
disclosed; the fresh bank is the clean cohort); cohort membership frozen
by construction, priors measured per site; n=1 per cell (one lineage,
one session — the standing lottery note); bars + P-x24a frozen at birth
before any compute.
"""
    (RD / "REPORT.md").write_text(report, encoding="utf-8")

    METRICS["outputs"] = {
        "figure": f"runs/{NAME}/x24_prior_census.png",
        "metrics": f"runs/{NAME}/metrics.json",
        "report": f"runs/{NAME}/REPORT.md",
    }
    METRICS["status"] = "COMPLETE — adjudicated"
    write_partial("P9 COMPLETE (report + figure)")
    log(f"DONE — verdict {word} (host spread "
        f"{clause_eval['host_g0']['cohort_a_spread']:.2f}x, anchor "
        f"{clause_eval['host_g0']['anchor_lift']:.1f}x, rhos "
        f"{riders['spearman_logprior_loglift_host_g0']:+.2f}/"
        f"{riders['spearman_logprior_loglift_neutral']:+.2f})")


if __name__ == "__main__":
    main()
