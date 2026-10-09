"""E314 — THE TAIL SCALPEL (the unlearning program's last road:
surgical erase AT THE GEOMETRY). This docstring carries the registered
question + bars VERBATIM from the dispatch letter + every frozen
convention, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (verbatim): "three roads to selective unlearning were
named; composition failed (e312: the protectors resurrect), sequencing
failed (e313: the membrane remembers past death). THE LAST ROAD: erase
only the target's OWN room-overlap components — the sharpest instrument
the laws license (e306: the bearer is the room-overlap tail; e310:
room-overlap necessary). If FACT3's read rides its room-overlap tail
and the siblings' reads ride THEIRS (overlapping but not identical), a
subtraction targeted at FACT3's specific tail overlap might kill FACT3
while the siblings' overlapping-but-distinct components survive."

THE DESIGN (verbatim): "on the five-family organism (its committed
state), compute FACT3's write delta (its own install's dW = its
inst_resume state minus the pre-install base — recoverable from the
committed checkpoints; disclose the reconstruction); project that delta
onto the SHARED room's basis; SUBTRACT a fraction alpha of FACT3's
room-overlap component from the organism (theta <- theta - alpha *
P_room(dW_FACT3)); sweep alpha in {0.25, 0.5, 1.0, 1.5}; read all five
facts at each alpha (one probe per alpha — cheap). The gradient-free
scalpel: pure geometry, no war."

THE ARMS (verbatim): "(a) THE SCALPEL (the sweep above); (b) the SHAM
control (subtract the same-norm projection of a RANDOM direction in the
room — does ANY in-room displacement of this size kill FACT3, or only
ITS OWN tail?)."

FROZEN BARS (verbatim):
  - SURGICAL-GEOMETRY: "some alpha kills FACT3 (< 0.01) while >= 3 of 4
    siblings hold (>= 0.5x baseline) AND the sham does NOT kill FACT3 —
    the first selective unlearning, bought at the geometry."
  - EVERYTHING-IS-SHARED: "the scalpel kills FACT3 but the siblings
    fall with it (their tails overlap FACT3's too much — the
    indivisibility is IN THE TAILS)."
  - NOTHING-DIES: "no alpha kills FACT3 even at 1.5x (the read's
    bearer is not FACT3's own tail — the bearer law's family reading
    needs revision)."
  - MIXED: "anything else — the alpha table verbatim (this is a
    dose-response — the TABLE is the deliverable even in MIXED)."

==== THE FROZEN CONVENTIONS (picked + frozen HERE at birth) ============

* THE ORGANISM := e291's committed five-fact family organism, LOADED
  BIT-EXACT (runs/checkpoints/e291_organism.pt; ckpt-md5 + flat-md5 +
  the behavioral five-fact panel g0 2e-6 / gm12 1e-5 vs the committed
  baselines — e294's G_FACTLOAD convention VERBATIM; the installs are
  NOT re-run: extend, don't repeat). THE SELECTIVITY TEST IS
  WITHIN-FAMILY (e291/e294's standing disclosure: five 12-window groups
  ALL bound to the name ZEPHYRA, five exactly-orthogonal 10k rooms in
  ONE shared frame; the siblings share representation (T270) — the
  hardest selectivity test the protocol can construct).
* THE ROOM := FACT3's OWN room (room index 3 of the five shared-frame
  rooms; the dispatch's "the target's OWN room-overlap components").
  The five rooms rebuilt from e291's frozen seeds (D 29111 / perm
  29112) + certified + BIT-gated vs e291_rooms.pt (e294's G_ROOMS5
  form). P_room := the room's EXACT SRCT projector (fp64 CPU pocketfft,
  e261's form); P_room(dW) the room-overlap component (the e306/e310
  bearer-law object). Co-reported (never a bar): dW_FACT3's per-room
  spectrum across ALL five rooms (the "overlapping but not identical"
  datum) + the union projection.
* THE RECONSTRUCTION (disclosed, gated): dW_FACT3 := flat(e291_install_
  F3_resume.pt s400) - flat(e291_install_F2_resume.pt s400) — FACT3's
  own install write, its pre-install base := FACT2's install end state
  (e291's serial chain base -> F1 -> ... -> F5 -> organism; e294's
  F3WRITE convention VERBATIM). GATED against e294's committed
  literals (norm 7.474247972268623; in-own-room 6.956357297542996;
  frac 0.9307099956213476) at 1e-9. THE FAMILY BAND: all five installs'
  dW (F1 := F1_resume - e001 base; Fi := Fi_resume - F(i-1)_resume)
  re-constructed, each's in-own-room share measured — FACT3's must sit
  in the family band (smoke check, live in full).
* THE SCALPEL (arm a): theta_alpha := theta_organism - alpha *
  P_room3(dW_FACT3), computed fp64 on the flat vector, cast fp32,
  probed. alpha in {0.25, 0.5, 1.0, 1.5} frozen (1.5 deliberately
  overshoots: the organism's room3 content 6.4806 < ||P_room3(dW_F3)||
  6.9564 — the later installs' AdamW weight-decay shrank the room
  after FACT3's install; disclosed, never a bar).
* THE SHAM (arm b): ONE fresh registered direction (this cell's seed
  31401, the family's per-cell rule .../29401/29402/HERE 31401): r ~
  N(0,1)^N fp64 -> p := P_room3(r) -> norm-matched to EXACTLY
  ||P_room3(dW_FACT3)||; theta_alpha := theta_organism - alpha * p.
  Same alphas, same probes. In-room fraction of p verified == 1 (the
  projector's residual, bar 1e-9); norm match bar 1e-12.
* THE READ := the five-fact g0 battery panel (e291/e294's own probe:
  12-window per-fact slices, mean p(Z) at the last position, CPU
  TinyGPT, bs 30) at every alpha for BOTH arms + the organism's own
  baseline panel (alpha := 0, the arms' shared origin). gm12 co-reported
  at every alpha, NEVER a bar (the family's convention). ONE panel per
  alpha per arm (the dispatch's "one probe per alpha").
* THE ADJUDICATION (frozen): kill_set := {alpha : scalpel_g0[alpha]
  [FACT3] < 0.01}; alpha_kill := the SMALLEST killing alpha (the
  dose-response's first kill; the table discloses any non-monotonicity)
  — "or the best alpha if none kills" := argmin FACT3 scalpel read, for
  the table's highlight only. hold_count(alpha) := #{siblings i :
  read_i(alpha) >= 0.5 x baseline_i} (baseline := the loaded organism's
  own committed panel; no twin — this cell has NO traffic, the passive
  decay confound does not exist by construction). NOTHING-DIES iff
  kill_set empty; else SURGICAL-GEOMETRY iff hold_count(alpha_kill)
  >= 3 AND sham_g0[alpha_kill][FACT3] >= 0.01 (the sham read at THE
  SAME alpha); else EVERYTHING-IS-SHARED iff hold_count(alpha_kill)
  <= 2; else MIXED (incl. the siblings-hold-but-sham-kills-too case —
  the honest "any in-room displacement of this size kills" reading).
  The full alpha table shown verbatim in every outcome.
* THE PROJECTOR IDENTITY (live in smoke AND full): on dW_FACT3 + on
  fresh probes — idempotency ||P(Px)-Px||/||Px||, complement
  orthogonality <Px, x-Px>, energy conservation ||Px||^2 + ||x-Px||^2
  == ||x||^2 — bars 1e-9 (e310's G_ORTH convention; observed ~1e-14).
* HARD GATES (a failure HALTs): {G_NAMEFREE, G_SPLICE, G_BATTERY,
  G_FACTSPLIT, G_FLATBASIS, G_PARENTS, G_BASE, G_VMBIND, G_SPANBIND,
  G_PROJ (the certification), G_ROOMS5 (incl. the bit-bind),
  G_FACTLOAD (the three-way organism gate), G_RECON (the dW_F3
  reconstruction vs e294's committed literals + the family band),
  G_SHAM (the sham's in-room + norm-match construction)}.
* COMPUTE: PURE GEOMETRY + PROBES — NO TRAINING, NO GRADIENTS, NO
  WAR. Projections fp64 CPU (pocketfft workers 2), probes CPU TinyGPT
  threads 4: CPU-ONLY (e306/e310's desk convention — disclosed; the
  GPU lane never taken, idle polls logged to runs/_envelope_log.jsonl
  tagged e314:<phase> as documentation). The dispatch's GPU envelope
  (bursts <= 180s, cooldowns, 85C line) is thereby vacuously satisfied
  — no GPU compute exists in this cell. TIMESTAMPS:
  datetime.now(UTC) only.
* Outputs: runs/e314/{metrics.json (PROGRESSIVE),
  e314_tail_scalpel.png, REPORT.md (executor-written), run.log
  (gitignored)}; the vectors artifact runs/e314/e314_tail_vectors.pt
  (gitignored *.pt; md5s in metrics). NO NOTES/THINKING/QUEUE/STATE
  edits (dispatch; the coordinator folds). Commit + push per phase.
* Smoke (E314_SMOKE=1): the full gate path (loads, rooms, bit-binds,
  reconstruction, projector identity, sham construction) LIVE at full
  k=10k (the CPU cell can afford the real rooms — the bit-bind is NOT
  vacuous), the sweep at alpha={0.5} only, own smoke dir
  (runs/e314_smoke/), SMOKE stamp on every read; NOTHING adjudicated.

REGISTERED PREDICTIONS (the executor's, frozen at birth):
  - P-e314a_tail_is_the_bearer: scalpel alpha >= 1.0 collapses FACT3
    below 0.01 — the closed bearer law (e306 sufficiency + e310
    necessity) projects onto the family: FACT3's read rides room3's
    install tail.
  - P-e314b_shared_read_path_bleeds: the siblings fall substantially
    at alpha >= 1.0 (>= 2 of 4 below 0.5x) — T270's one-memoria law
    survives spectral disjointness (the supports are disjoint but the
    forward path is one) -> EVERYTHING-IS-SHARED.
  - P-e314c_sham_kills_too: the sham ALSO kills FACT3 at alpha >= 1.0
    — ||P_room3(dW_F3)|| = 6.96 sits ~300x above the 0.3233% transport
    bracket; a same-norm in-room scramble is not tail-specific at this
    dose -> MIXED via the sham clause wherever P-b fails.
  - P-e314d_nothing: FACT3 survives even 1.5x (the controller era
    reshaped room3 away from dW_F3's tail; the bearer law's family
    reading revisited) -> NOTHING-DIES.

Run:  cd lab && python e314_tail_scalpel.py    (E314_SMOKE=1 shakedown)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
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
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402 — the SRCT
                                                      # transform (sf/DCT)

torch.set_num_threads(4)           # CPU probing lane (shared machine)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E314_SMOKE") == "1"
NAME = "e314_smoke" if SMOKE else "e314"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def gpu_idle_poll(tag: str) -> dict:
    """Informational envelope poll (the lane is never taken; documented)."""
    s = common.gpu_status()
    common._log_envelope_poll(f"{NAME}:{tag}", s["util"], s["temp"],
                              s["temp"] < E261.TEMP_HARD)
    return s


# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (the chain origin)
BASE_CK_MD5 = "d114536d1c0983ab3be67f67ff0667c8"
VMAP_CK = "e258_vmap.pt"          # e258's committed v-map (the rooms' probe)
VMAP_CK_MD5 = "baafd327e9e35cffa2d41ea4ecf5fc5c"
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span
SPAN_CK_MD5 = "3464ff6081d8e402103dd9ca65bfb53a"
CKPT_DIR = GB.CKPT_DIR

ORG_CK = "e291_organism.pt"       # e291's committed five-fact organism
ORG_CK_MD5 = "ee2bad6be9f55fd94ebf3a367967da30"
ORG_FLAT_MD5 = "876907be13dc0f08412d5ffa36e4d57a"
ROOMS291_CK = "e291_rooms.pt"     # e291's committed rooms artifact (the bind)
ROOMS291_CK_MD5 = "163f90a1edcfbd1dbc44bfd72296d5cb"

# the five install resume checkpoints (e291's serial chain, all s400)
INST_CKS = {
    "FACT1": ("e291_install_F1_resume.pt", "fc31425fb7cbd2cee213fe7503669aff"),
    "FACT2": ("e291_install_F2_resume.pt", "8b6b0ba384a2b47261569c2c04d22aaf"),
    "FACT3": ("e291_install_F3_resume.pt", "51c90f35771c5876dafcd5006e2c06d9"),
    "FACT4": ("e291_install_F4_resume.pt", "dc06875752898e6f93ee4159ea06b8d3"),
    "FACT5": ("e291_install_F5_resume.pt", "2db9ca37b9c420033708259785002eb3"),
}

# e294's committed F3WRITE literals (THE reconstruction's bind)
E294_F3WRITE_NORM = 7.474247972268623
E294_F3WRITE_IN_OWN_ROOM = 6.956357297542996
E294_F3WRITE_IN_OWN_FRAC = 0.9307099956213476
E294_RECON_TOL = 1e-9

# ---- the parents, HARD-BOUND (md5s read at runtime; Rule 12) ------------
E291_METRICS = E43.REPO / "runs" / "e291" / "metrics.json"
E291_MD5 = "fc9cb859f2ef2603397c971dbcf5c618"
E291_VERDICT = "PEACEFUL-COEXISTENCE"

E294_METRICS = E43.REPO / "runs" / "e294" / "metrics.json"
E294_MD5 = "6ffa887b2ab8b50f91124d31cbc4b735"
E294_VERDICT = "COLLATERAL"

E306_METRICS = E43.REPO / "runs" / "e306" / "metrics.json"
E306_MD5 = "a4f7a55ae15f6db0f809352aa182d6ff"
E306_VERDICT = "READS-CLIFF"

E310_METRICS = E43.REPO / "runs" / "e310" / "metrics.json"
E310_MD5 = "713fd5c01f1b946ed7148b41a837dc58"
E310_VERDICT = "READ-DEAD-MASS-HIGH"

E312_METRICS = E43.REPO / "runs" / "e312" / "metrics.json"
E312_MD5 = "e970738ac6405ec31cab572ed0dc2b9d"
E312_VERDICT = "MIXED"

E313_METRICS = E43.REPO / "runs" / "e313" / "metrics.json"
E313_MD5 = "4894788c7f3a56082ce7cb41acd64b3f"

# ---- e291's committed baselines (the family's own frozen table) ----------
E291_BASELINES_G0 = {
    "FACT1": 0.26670998334884644,
    "FACT2": 0.36356988549232483,
    "FACT3": 0.2677536904811859,
    "FACT4": 0.28865179419517517,
    "FACT5": 0.3815295398235321,
}
E291_BASELINES_GM12 = {
    "FACT1": 0.10490661859512329,
    "FACT2": 0.13552233524854597,
    "FACT3": 0.08752423524856567,
    "FACT4": 0.11411947747005692,
    "FACT5": 0.14677566289901733,
}

# ---- the rooms / the family ----------------------------------------------
ROOM_K = 10_000                                  # full k in smoke too (CPU)
N_ROOMS = 5
ROOM_D_SEED = 29111
ROOM_S_SEED = 29112
FACTS = tuple(f"FACT{i}" for i in range(1, N_ROOMS + 1))
TARGET_FACT = "FACT3"
TARGET_IDX = 2
GROUP_SIZE = 12

# ---- the sweep + the sham (frozen) ---------------------------------------
SWEEP = (0.25, 0.5, 1.0, 1.5)
SMOKE_SWEEP = (0.5,)
# e290's committed realized-drift bracket (the passive transport price)
E290_DRIFT_BRACKET = (0.0007099904808640174, 0.0032331212727527057)
SHAM_SEED = 31401                    # this cell's ONE fresh registered seed
ERASE_BAR = 0.01                     # FACT3 < this := KILLED (absolute)
HOLD_FRAC = 0.5                      # a sibling HOLDS: >= 0.5x its baseline
SIBLING_MIN_HOLD = 3                 # ">= 3 of 4 siblings"
FACT_READ_TOL_G0 = 2e-6              # the family's cross-session read law
FACT_READ_TOL_GM12 = 1e-5
PROJ_ID_BAR = 1e-9                   # the projector-identity bar (e310's)
SHAM_NORM_TOL = 1e-12
VOCAB_EXPECT = 65

ARMS = ("SCALPEL", "SHAM")

REGISTERED = {
    "question_verbatim": (
        "three roads to selective unlearning were named; composition "
        "failed (e312: the protectors resurrect), sequencing failed "
        "(e313: the membrane remembers past death). THE LAST ROAD: erase "
        "only the target's OWN room-overlap components — the sharpest "
        "instrument the laws license (e306: the bearer is the room-"
        "overlap tail; e310: room-overlap necessary). If FACT3's read "
        "rides its room-overlap tail and the siblings' reads ride THEIRS "
        "(overlapping but not identical), a subtraction targeted at "
        "FACT3's specific tail overlap might kill FACT3 while the "
        "siblings' overlapping-but-distinct components survive."),
    "design_verbatim": (
        "on the five-family organism (its committed state), compute "
        "FACT3's write delta (its own install's dW = its inst_resume "
        "state minus the pre-install base — recoverable from the "
        "committed checkpoints; disclose the reconstruction); project "
        "that delta onto the SHARED room's basis; SUBTRACT a fraction "
        "alpha of FACT3's room-overlap component from the organism "
        "(theta <- theta - alpha * P_room(dW_FACT3)); sweep alpha in "
        "{0.25, 0.5, 1.0, 1.5}; read all five facts at each alpha (one "
        "probe per alpha — cheap). The gradient-free scalpel: pure "
        "geometry, no war."),
    "arms_verbatim": {
        "SCALPEL": "(a) THE SCALPEL (the sweep above)",
        "SHAM": ("(b) the SHAM control (subtract the same-norm projection "
                 "of a RANDOM direction in the room — does ANY in-room "
                 "displacement of this size kill FACT3, or only ITS OWN "
                 "tail?)"),
    },
    "bars_verbatim": {
        "SURGICAL-GEOMETRY": ("some alpha kills FACT3 (< 0.01) while >= 3 "
                              "of 4 siblings hold (>= 0.5x baseline) AND "
                              "the sham does NOT kill FACT3 — the first "
                              "selective unlearning, bought at the "
                              "geometry."),
        "EVERYTHING-IS-SHARED": ("the scalpel kills FACT3 but the "
                                 "siblings fall with it (their tails "
                                 "overlap FACT3's too much — the "
                                 "indivisibility is IN THE TAILS)."),
        "NOTHING-DIES": ("no alpha kills FACT3 even at 1.5x (the read's "
                         "bearer is not FACT3's own tail — the bearer "
                         "law's family reading needs revision)."),
        "MIXED": ("anything else — the alpha table verbatim (this is a "
                  "dose-response — the TABLE is the deliverable even in "
                  "MIXED)."),
    },
    "registration": ("question + design + arms + bars VERBATIM from the "
                     "dispatch letter; every convention picked + frozen "
                     "HERE at birth BEFORE compute; this script committed "
                     "at birth; adjudicate against exactly this; no bar "
                     "shopping."),
    "predictions": {
        "P-e314a_tail_is_the_bearer": (
            "scalpel alpha >= 1.0 collapses FACT3 below 0.01 — the closed "
            "bearer law projects onto the family: FACT3's read rides "
            "room3's install tail."),
        "P-e314b_shared_read_path_bleeds": (
            "the siblings fall substantially at alpha >= 1.0 (>= 2 of 4 "
            "below 0.5x) — T270's one-memoria law survives spectral "
            "disjointness -> EVERYTHING-IS-SHARED."),
        "P-e314c_sham_kills_too": (
            "the sham ALSO kills FACT3 at alpha >= 1.0 — 6.96 sits ~300x "
            "above the transport bracket; a same-norm in-room scramble "
            "is not tail-specific at this dose -> MIXED via the sham "
            "clause wherever P-b fails."),
        "P-e314d_nothing": (
            "FACT3 survives even 1.5x (the controller era reshaped room3 "
            "away from dW_F3's tail) -> NOTHING-DIES."),
    },
    "family_disclosure": (
        "THE SELECTIVITY TEST IS WITHIN-FAMILY: the vehicle is e291's "
        "five-fact FAMILY organism (five 12-window context groups ALL "
        "bound to the SAME name, five exactly-orthogonal 10k rooms in "
        "one shared frame); the siblings SHARE representation (T270), so "
        "collateral is the expected failure mode and a clean erase the "
        "strongest form of the result."),
}

deviations: list[str] = [
    "PURE GEOMETRY + PROBES, NO TRAINING (the dispatch's own design: "
    "'the gradient-free scalpel: pure geometry, no war') — no corpus "
    "traffic, no optimizers, no gradients; the CPU-only compute "
    "convention of e306/e310 (disclosed; the GPU lane never taken — "
    "idle envelope polls logged as documentation, tags e314:<phase>).",
    "THE ROOM RESOLVED TO FACT3'S OWN ROOM (room index 3 of the five "
    "shared-frame rooms): the dispatch's 'the target's OWN room-overlap "
    "components' + 'P_room(dW_FACT3)' — the family analog of e306/"
    "e310's fact-vs-its-own-room object. The dispatch's phrase 'the "
    "SHARED room's basis' read as the family's shared-frame room "
    "structure (e291's five rooms share ONE frame; the rooms "
    "themselves are exactly-orthogonal). Co-reported never a bar: "
    "dW_F3's per-room spectrum across all five rooms + the union "
    "projection.",
    "THE RECONSTRUCTION'S PRE-INSTALL BASE := FACT2's install end state "
    "(e291's serial chain: base -> F1 -> ... -> F5 -> organism; the F1 "
    "install's origin := the e001 base) — e294's F3WRITE convention "
    "VERBATIM, gated against e294's committed literals at 1e-9. "
    "DISCLOSED: the install-time dW_F3 is NOT bit-identical to the "
    "ORGANISM's current room3 content (||P_room3(dW_F3)|| 6.9564 vs "
    "the organism's room3 norm 6.4806 — the later installs' AdamW "
    "weight decay (wd 0.1, decoupled, unprojected) shrank room3 after "
    "FACT3's install); alpha 1.5 deliberately overshoots into the "
    "sign-flipped regime — the dispatch's own dial, never a bar.",
    "THE ADJUDICATION AT THE FIRST KILLING ALPHA (the frozen "
    "operationalization of 'the verdict at the alpha that kills FACT3 "
    "< 0.01'): alpha_kill := min(kill_set); the sham clause read at THE "
    "SAME alpha; the sibling-hold count absolute (>= 0.5 x the loaded "
    "organism's own committed baseline — no twin exists: zero traffic "
    "means zero passive decay by construction); 'the best alpha if "
    "none kills' := argmin FACT3 scalpel read (table highlight only).",
    "THE gm12 CO-REPORT at every alpha (the family's convention: gm12 "
    "never a bar) — the dispatch's 'one probe per alpha' := one "
    "five-fact PANEL per alpha (g0 the bar + gm12 the co-report, both "
    "in the same pass).",
    "NO NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke (E314_SMOKE=1): the full gate path LIVE at real k=10k (loads, "
    "rooms + the bit-bind NOT vacuous, the reconstruction vs e294's "
    "literals, the projector identity, the sham construction), the "
    "sweep at alpha={0.5} only, own smoke dir; NOTHING adjudicated "
    "(SMOKE stamp on every read).",
    "n=1 per arm-point, one lineage, one session (the g-series standing "
    "caveat); the dose-response TABLE is the registered object, not any "
    "single point.",
]

# ------------------------------------------------------------------ helpers
def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(E43.REPO), capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def git_commit_push(msg: str, paths: list[str]) -> str:
    """Commit ONLY this cell's paths; push to origin main."""
    try:
        subprocess.run(["git", "add", *paths], cwd=str(E43.REPO),
                       check=True, timeout=60)
        r = subprocess.run(["git", "commit", "-m", msg], cwd=str(E43.REPO),
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            log(f"[git] commit skipped: {r.stdout.strip()[:120]} "
                f"{r.stderr.strip()[:120]}")
            return git_head()
        subprocess.run(["git", "push", "origin", "main"], cwd=str(E43.REPO),
                       capture_output=True, text=True, timeout=120)
        h = git_head()
        log(f"[git] committed + pushed {h[:12]}: {msg[:90]}")
        return h
    except Exception as e:                                  # noqa: BLE001
        log(f"[git] ERROR: {e}")
        return git_head()


metrics: dict = {"experiment": "e314_tail_scalpel",
                 "date": common.now_iso(),
                 "status": "PARTIAL: startup (bars registered at birth)",
                 "smoke": SMOKE,
                 "birth_commit": git_head(),
                 "registration": REGISTERED["registration"],
                 "registered": REGISTERED,
                 "cpu_only": True,
                 "threads": {"torch": 4, "pocketfft": E261.DCT_WORKERS},
                 "envelope": {
                     "device": "CPU ONLY (pure geometry + CPU probes — no "
                               "training, no gradients; e306/e310's desk "
                               "convention)",
                     "gpu_note": "the GPU lane never taken; idle polls "
                                 "logged to runs/_envelope_log.jsonl "
                                 "tagged e314:<phase> as documentation — "
                                 "the dispatch's burst/cooldown/85C "
                                 "envelope vacuously satisfied",
                     "timestamps": "datetime.now(UTC) only"},
                 "deviations": deviations,
                 "gates": {}}
METRICS_PATH = RD / "metrics.json"


def write_partial(note: str) -> None:
    metrics["phase_note"] = note
    metrics["date_updated"] = common.now_iso()
    save_json(METRICS_PATH, metrics)
    log(f"[metrics] partial saved ({note})")


# ======================================================================
# THE FIVE ROOMS (e291's SharedFrameRooms PORTED WHOLE from e294 — the
# family's own instrument; the committed files NOT modified)
# ======================================================================
class SharedFrameRooms:
    """Five rank-k SRCT rooms as DISJOINT spectral supports of ONE frame:
    ONE seeded +-1 diagonal D and ONE seeded permutation split into five
    disjoint sorted index sets S_i (k each). Room i's exact projector:
        P_i x = D . idct(mask_{S_i}(dct(D . x)))
    (fp64 pocketfft, CPU — e261's SRCT form verbatim, the frame shared).
    The pairwise projectors compose to ZERO exactly (disjoint supports),
    and the union P_U = sum_i P_i is ONE exact projector with the 50k
    union mask."""

    def __init__(self, n: int, k: int, n_rooms: int, seed_d: int,
                 seed_s: int, v_flat64: np.ndarray, Vp64: np.ndarray,
                 params_ref, dev):
        assert 0 < k * n_rooms <= n
        self.n, self.k, self.n_rooms = int(n), int(k), int(n_rooms)
        self.dev = dev
        g = torch.Generator().manual_seed(seed_d)
        self.D = torch.where(torch.rand(n, generator=g) < 0.5,
                             -1.0, 1.0).numpy().astype(np.float64)
        g2 = torch.Generator().manual_seed(seed_s)
        perm = torch.randperm(n, generator=g2)[:self.k * self.n_rooms]
        self.sets = [perm[i * k:(i + 1) * k].sort().values.numpy()
                     for i in range(self.n_rooms)]
        all_idx = np.concatenate(self.sets)
        assert len(set(all_idx.tolist())) == self.k * self.n_rooms
        self.masks = []
        for S in self.sets:
            m = np.zeros(n, dtype=np.float64)
            m[S] = 1.0
            self.masks.append(m)
        self.mask_union = np.zeros(n, dtype=np.float64)
        self.mask_union[all_idx] = 1.0
        self.seed_d, self.seed_s = seed_d, seed_s
        self.v64 = v_flat64.astype(np.float64)
        self.mean_v = float(self.v64.mean())
        self.Vp = Vp64.astype(np.float64)
        self.r_span = int(Vp64.shape[0])
        self.offsets, self.shapes = [], []
        off = 0
        for p in params_ref:
            self.offsets.append((off, off + p.numel()))
            self.shapes.append(tuple(p.shape))
            off += p.numel()
        assert off == self.n, f"flat size {off} != {self.n}"

    def coeffs(self, x64: np.ndarray) -> np.ndarray:
        return E261.sf.dct(self.D * x64, type=2, norm="ortho",
                           workers=E261.DCT_WORKERS)

    def recon(self, c_masked: np.ndarray) -> np.ndarray:
        return self.D * E261.sf.idct(c_masked, type=2, norm="ortho",
                                     workers=E261.DCT_WORKERS)

    def project_masked(self, x64: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.recon(self.coeffs(x64) * mask)

    def project_room(self, x64: np.ndarray, i: int) -> np.ndarray:
        return self.project_masked(x64, self.masks[i])

    def project_union(self, x64: np.ndarray) -> np.ndarray:
        return self.project_masked(x64, self.mask_union)

    def room_fracs_from_coeffs(self, c: np.ndarray) -> list:
        tot = float(np.sqrt((c * c).sum()))
        if tot <= 0:
            return [0.0] * self.n_rooms
        return [float(np.sqrt((c[S] * c[S]).sum())) / tot for S in self.sets]

    def in_room_fracs(self, x64: np.ndarray) -> list:
        return self.room_fracs_from_coeffs(self.coeffs(x64))

    def union_frac(self, x64: np.ndarray) -> float:
        c = self.coeffs(x64)
        tot = float(np.sqrt((c * c).sum()))
        if tot <= 0:
            return 0.0
        cu = c * self.mask_union
        return float(np.sqrt((cu * cu).sum()) / tot)

    def certify(self, probes: int = E261.CERT_PROBES,
                seed: int = E261.CERT_SEED) -> dict:
        rng = np.random.default_rng(seed)
        out = {"probes": probes, "seed": seed}
        x = rng.standard_normal(self.n)
        xr = self.recon(self.coeffs(x))
        out["dct_roundtrip_rel"] = float(np.linalg.norm(xr - x)
                                         / np.linalg.norm(x))
        per_room = {}
        for i in range(self.n_rooms):
            idem, kept2 = [], []
            for _ in range(probes):
                y = rng.standard_normal(self.n)
                py = self.project_room(y, i)
                ppy = self.project_room(py, i)
                idem.append(float(np.linalg.norm(ppy - py)
                                  / np.linalg.norm(py)))
                kept2.append(float((py @ py) / (y @ y)))
            ovr = [float(np.linalg.norm(self.project_room(self.Vp[j], i)))
                   for j in range(self.r_span)]
            bar = max(5.0 * math.sqrt(2.0 * self.k) / self.n, 1e-9)
            per_room[f"room{i + 1}"] = {
                "k": self.k, "seeds": [self.seed_d, self.seed_s],
                "idempotency_max": max(idem),
                "kept2_mean": float(np.mean(kept2)),
                "kept2_expect": self.k / self.n,
                "kept2_bar_10sig": bar,
                "kept2_pass": bool(abs(float(np.mean(kept2)) - self.k / self.n)
                                   <= bar),
                "idempotency_pass": bool(max(idem) <= 1e-8),
                "span_overlap_mean": float(np.mean(ovr)),
                "span_overlap_expect": math.sqrt(self.k / self.n)}
        kept2u = []
        for _ in range(probes):
            y = rng.standard_normal(self.n)
            pu = self.project_union(y)
            kept2u.append(float((pu @ pu) / (y @ y)))
        out["union_kept2_mean"] = float(np.mean(kept2u))
        out["union_kept2_expect"] = self.n_rooms * self.k / self.n
        out["union_kept2_pass"] = bool(
            abs(float(np.mean(kept2u)) - self.n_rooms * self.k / self.n)
            <= max(5.0 * math.sqrt(2.0 * self.n_rooms * self.k) / self.n,
                   1e-9))
        pairwise = np.zeros((self.n_rooms, self.n_rooms))
        for _ in range(probes):
            y = rng.standard_normal(self.n)
            for i in range(self.n_rooms):
                pi = self.project_room(y, i)
                for j in range(self.n_rooms):
                    if i != j:
                        pj = self.project_room(pi, j)
                        pairwise[i, j] = max(
                            pairwise[i, j],
                            float(np.linalg.norm(pj)
                                  / max(np.linalg.norm(pi), 1e-30)))
        out["pairwise_proj_max"] = pairwise.tolist()
        out["pairwise_zero_bar"] = 1e-12
        out["pairwise_pass"] = bool(pairwise.max() <= 1e-12)
        out["per_room"] = per_room
        out["pass"] = bool(out["dct_roundtrip_rel"] <= 1e-8
                           and all(r["kept2_pass"] and r["idempotency_pass"]
                                   for r in per_room.values())
                           and out["union_kept2_pass"]
                           and out["pairwise_pass"])
        return out


# ======================================================================
def apply_flat(net, flat32: torch.Tensor, offsets, shapes) -> None:
    with torch.no_grad():
        for p, (a, b), shp in zip(net.parameters(), offsets, shapes):
            p.copy_(flat32[a:b].reshape(shp))


def main():
    sweep = SMOKE_SWEEP if SMOKE else SWEEP
    metrics.update({
        "phase": ("THE TAIL SCALPEL (the last road): subtract alpha x "
                  "P_room3(dW_FACT3) — the target's OWN room-overlap tail "
                  "(e306/e310's closed bearer law's target list) — from "
                  "the committed five-family organism at alpha "
                  f"{list(sweep)}; the SHAM control (a same-norm RANDOM "
                  "in-room direction, seed 31401); the five-fact g0 panel "
                  "at every alpha: SURGICAL-GEOMETRY / EVERYTHING-IS-"
                  "SHARED / NOTHING-DIES / MIXED"),
        "sweep": list(sweep),
    })
    log(f"E314 — THE TAIL SCALPEL (smoke={SMOKE}) -> {RD}")
    log(f"the bars: SURGICAL-GEOMETRY (kill FACT3 < {ERASE_BAR} + >= "
        f"{SIBLING_MIN_HOLD} of 4 siblings >= {HOLD_FRAC:.0%}x baseline + "
        "the sham does NOT kill) / EVERYTHING-IS-SHARED (the scalpel "
        "kills, the siblings fall) / NOTHING-DIES (no alpha kills even at "
        "1.5x) / MIXED (the table verbatim)")
    write_partial("startup (bars + arms + the scalpel's law registered, "
                  "committed at birth)")
    set_seed(SHAM_SEED)       # global init only; every RNG is its own

    # ================= P0: the protocol rebuild (the probe's gates) =====
    gpu_idle_poll("P0:protocol")
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    vocab = corpus.vocab_size
    zid = stoi["Z"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)
    assert vocab == VOCAB_EXPECT, f"vocab drift: {vocab}"

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
    G_SPLICE = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE])}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]

    g0_ids_f = {f"FACT{i + 1}": g0_ids[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
                for i in range(N_ROOMS)}
    gm12_ids_f = {f"FACT{i + 1}":
                  gm12_ids[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
                  for i in range(N_ROOMS)}
    G_FACTSPLIT = {
        "form": "e291's frozen five-fact split (12 windows each, disjoint "
                "cover); the probe batteries sliced the same way",
        "disjoint_cover": bool(sum(len(v) for v in g0_ids_f.values()) == 60),
        "battery_shapes": {k: list(v.shape) for k, v in g0_ids_f.items()},
        "pass": bool(all(list(v.shape) == [GROUP_SIZE, G1.PRE]
                         for v in g0_ids_f.values())),
    }
    assert G_FACTSPLIT["pass"], "fact split gate FAILED"
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY,
                             "G_FACTSPLIT": G_FACTSPLIT})
    log("P0: protocol gates PASS (namefree / splice 19+41 / battery shapes "
        "/ the five-fact split)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e291m = json.loads(E291_METRICS.read_text(encoding="utf-8"))
    e294m = json.loads(E294_METRICS.read_text(encoding="utf-8"))
    e306m = json.loads(E306_METRICS.read_text(encoding="utf-8"))
    e310m = json.loads(E310_METRICS.read_text(encoding="utf-8"))
    e312m = json.loads(E312_METRICS.read_text(encoding="utf-8"))
    e313m = json.loads(E313_METRICS.read_text(encoding="utf-8"))
    e294_f3w = e294m["gates"]["G_F3WRITE"]
    G_PARENTS = {
        "e291_metrics": {"path": str(E291_METRICS), "md5": md5of(E291_METRICS),
                         "bound_md5": E291_MD5,
                         "verdict": e291m["adjudication"]["verdict"],
                         "note": "THE VEHICLE'S PARENT (the family "
                                 "organism + rooms + baselines)"},
        "e294_metrics": {"path": str(E294_METRICS), "md5": md5of(E294_METRICS),
                         "bound_md5": E294_MD5,
                         "verdict": e294m["adjudication"]["verdict"],
                         "f3write_norm": e294_f3w["f3_write_norm"],
                         "f3write_in_own_room": e294_f3w[
                             "f3_write_in_own_room_norm"],
                         "note": "THE RECONSTRUCTION'S BIND (F3WRITE "
                                 "committed literals) + the anti's "
                                 "COLLATERAL (road 0: unprotected erase "
                                 "kills all five)"},
        "e306_metrics": {"path": str(E306_METRICS), "md5": md5of(E306_METRICS),
                         "bound_md5": E306_MD5,
                         "verdict": e306m["verdict"]["word"],
                         "note": "the bearer law's sufficiency half "
                                 "(MASS-IS-NOT-MEMORY; the read tracks the "
                                 "room-overlap energy)"},
        "e310_metrics": {"path": str(E310_METRICS), "md5": md5of(E310_METRICS),
                         "bound_md5": E310_MD5,
                         "verdict": e310m["verdict"]["word"],
                         "note": "the bearer law's necessity half "
                                 "(READ-DEAD-MASS-HIGH; the G_ORTH "
                                 "projector-identity convention)"},
        "e312_metrics": {"path": str(E312_METRICS), "md5": md5of(E312_METRICS),
                         "bound_md5": E312_MD5,
                         "verdict": e312m["adjudication"]["verdict"],
                         "note": "road 1 (composition): the protectors "
                                 "resurrect the target"},
        "e313_metrics": {"path": str(E313_METRICS), "md5": md5of(E313_METRICS),
                         "bound_md5": E313_MD5,
                         "verdict_A": e313m["adjudication_A"]["verdict"],
                         "verdict_B": e313m["adjudication_B"]["verdict"],
                         "note": "road 2 (sequencing): SEQ-LEAKY — the "
                                 "membrane remembers past death"},
        "pass": bool(md5of(E291_METRICS) == E291_MD5
                     and md5of(E294_METRICS) == E294_MD5
                     and md5of(E306_METRICS) == E306_MD5
                     and md5of(E310_METRICS) == E310_MD5
                     and md5of(E312_METRICS) == E312_MD5
                     and md5of(E313_METRICS) == E313_MD5
                     and e291m["adjudication"]["verdict"] == E291_VERDICT
                     and e294m["adjudication"]["verdict"] == E294_VERDICT
                     and e306m["verdict"]["word"] == E306_VERDICT
                     and e310m["verdict"]["word"] == E310_VERDICT
                     and e312m["adjudication"]["verdict"] == E312_VERDICT
                     and e313m["adjudication_B"]["verdict"] == "SEQ-LEAKY"),
    }
    assert G_PARENTS["pass"], "parent bind failed"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: G_PARENTS PASS — e291 PEACEFUL-COEXISTENCE / e294 COLLATERAL "
        "/ e306 READS-CLIFF / e310 READ-DEAD-MASS-HIGH / e312 MIXED / e313 "
        "SEQ-LEAKY — all md5-bound (the three roads' fates: composition "
        "fell, sequencing fell, THE GEOMETRY remains)")
    del e291m, e294m, e306m, e310m, e312m, e313m
    write_partial("P0b parents hard-bound")

    # ================= P0c: the base + the flat basis ====================
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    N = base_net.num_params()
    assert N == GB.G1B_PARAMS, f"base param count {N}"
    assert md5of(CKPT_DIR / BASE_CK) == BASE_CK_MD5, "base ckpt md5 drift"

    named_p = list(base_net.named_parameters())
    sd_keys = list(base_sd.keys())
    G_FLATBASIS = {
        "form": "the flat vector <-> state-dict mapping's safety (e306's "
                "G_FLATBASIS): parameter keys == state-dict keys in order, "
                "zero buffers",
        "params_count": len(named_p),
        "buffers_count": sum(1 for k in base_sd if k not in
                             dict(named_p)),
        "key_order_matches_parameters": bool(
            [k for k, _ in named_p] == sd_keys),
        "n_params": N,
        "pass": bool([k for k, _ in named_p] == sd_keys
                     and all(k in dict(named_p) for k in sd_keys)
                     and len(sd_keys) == len(named_p)),
    }
    assert G_FLATBASIS["pass"], f"flat basis FAILED: {G_FLATBASIS}"

    base_panel = {f: G1.battery_cell(base_net, ids, zid)["mean_pz"]
                  for f, ids in g0_ids_f.items()}
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "md5": md5of(CKPT_DIR / BASE_CK), "params": N,
              "fact_free_panel_g0": {k: float(v)
                                     for k, v in base_panel.items()},
              "pass": bool(all(v <= 0.05 for v in base_panel.values()))}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"].update({"G_FLATBASIS": G_FLATBASIS, "G_BASE": G_BASE})
    log(f"P0c: G-BASE + G-FLATBASIS PASS (fact-free panel max "
        f"{max(base_panel.values()):.2e}; {N} params, no buffers)")
    write_partial("P0c base + flat basis PASSED")

    base_flat_np = flat_params_cpu(G1.evl_load(base_sd)) \
        .double().numpy().astype(np.float64)

    # ---- the span + v-map binds (the rooms' probe instruments) ----------
    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    G_VMBIND = {"path": f"runs/checkpoints/{VMAP_CK}",
                "md5": md5of(CKPT_DIR / VMAP_CK), "bound_md5": VMAP_CK_MD5,
                "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
                "size": int(v_flat32.numel()), "expected_size": N,
                "pass": bool(vmap_art.get("meta", {}).get("experiment")
                             == "e258" and int(v_flat32.numel()) == N
                             and md5of(CKPT_DIR / VMAP_CK) == VMAP_CK_MD5)}
    assert G_VMBIND["pass"], "v-map bind failed"
    span_art = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / SPAN_CK),
                  "bound_md5": SPAN_CK_MD5,
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "pass": bool(md5of(CKPT_DIR / SPAN_CK) == SPAN_CK_MD5
                               and int(Vp.shape[1]) == N)}
    assert G_SPANBIND["pass"], "span bind failed"
    metrics["gates"].update({"G_VMBIND": G_VMBIND, "G_SPANBIND": G_SPANBIND})

    # ================= P1: THE FIVE ROOMS (rebuilt + certified + bound) ==
    gpu_idle_poll("P1:rooms")
    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = SharedFrameRooms(N, ROOM_K, N_ROOMS, ROOM_D_SEED, ROOM_S_SEED,
                             v_flat32.numpy().astype(np.float64),
                             Vp.numpy().astype(np.float64), params_ref,
                             torch.device("cpu"))
    cert = rooms.certify()
    G_PROJ = {
        "form": f"each room + the union certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the "
                "DCT roundtrip identity; IDEMPOTENCY + kept^2 rank; the "
                "union's kept^2 vs 5k/N",
        "reads": cert,
        "pass": bool(cert["pass"]),
    }
    assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"

    rooms_art = torch.load(CKPT_DIR / ROOMS291_CK, map_location="cpu",
                           weights_only=False)
    D_ok = bool(np.array_equal(np.asarray(rooms_art["model"]["D"]), rooms.D))
    sets_ok = all(np.array_equal(np.asarray(rooms_art["model"]["sets"][i]),
                                 rooms.sets[i]) for i in range(N_ROOMS))
    bit_bind = {
        "form": "the rebuilt frame BIT-COMPARED vs e291's committed rooms "
                "artifact (D bit-exact + all five index sets bit-exact — "
                "the vehicle's rooms are THE organism's own rooms); LIVE "
                "in smoke too (real k=10k — the CPU cell's disclosure)",
        "D_bit_equal": D_ok, "sets_bit_equal_all": sets_ok,
        "vacuous": False,
        "pass": bool(D_ok and sets_ok
                     and rooms_art["model"].get("k") == ROOM_K
                     and rooms_art["model"].get("n_rooms") == N_ROOMS),
    }
    G_ROOMS5 = {
        "form": "THE FIVE ROOMS' pairwise orthogonality (disjoint spectral "
                "supports — exactly zero) + THE BIT-BIND vs the committed "
                "e291 rooms",
        "pairwise_proj_max": cert["pairwise_proj_max"],
        "bar": 1e-12,
        "construction": {"seed_d": ROOM_D_SEED, "seed_s": ROOM_S_SEED,
                         "k": ROOM_K, "n_rooms": N_ROOMS},
        "bit_bind": bit_bind,
        "pass": bool(cert["pairwise_pass"] and bit_bind["pass"]),
    }
    assert G_ROOMS5["pass"], "five-room gate FAILED (pairwise or bit-bind)"
    metrics["gates"].update({"G_PROJ": G_PROJ, "G_ROOMS5": G_ROOMS5})
    pw = np.array(cert["pairwise_proj_max"])
    log(f"P1: G_PROJ + G_ROOMS5 PASS (pairwise max {pw.max():.1e} bar 1e-12; "
        f"BIT-BIND vs e291_rooms.pt D + 5 sets bit-exact; k={ROOM_K})")
    write_partial("P1 the five rooms rebuilt + certified + bit-bound")

    # ================= P2: THE ORGANISM (loaded + gated) ================
    gpu_idle_poll("P2:organism")
    org_art = torch.load(CKPT_DIR / ORG_CK, map_location="cpu",
                         weights_only=False)
    organism_sd = org_art["model"]
    org_net = G1.evl_load(organism_sd)
    org_flat = flat_params_cpu(org_net)
    org_flat_np = org_flat.double().numpy().astype(np.float64)
    org_flat_md5 = hashlib.md5(org_flat.numpy().tobytes()).hexdigest()
    baseline_panel = {f: G1.battery_cell(org_net, g0_ids_f[f], zid)["mean_pz"]
                      for f in FACTS}
    baseline_panel12 = {f: G1.battery_cell(org_net, gm12_ids_f[f],
                                           zid)["mean_pz"] for f in FACTS}
    baselines = {k: float(v) for k, v in baseline_panel.items()}
    panel_diffs = {k: abs(baselines[k] - E291_BASELINES_G0[k])
                   for k in baselines}
    panel12_diffs = {f: abs(float(baseline_panel12[f])
                            - E291_BASELINES_GM12[f]) for f in FACTS}
    G_FACTLOAD = {
        "form": "THE THREE-WAY ORGANISM GATE (e294's G_FACTLOAD VERBATIM): "
                "ckpt md5 + flat md5 + the behavioral panel (g0 2e-6 / "
                "gm12 1e-5 vs e291's committed baselines)",
        "checkpoint": f"runs/checkpoints/{ORG_CK}",
        "ckpt_md5": md5of(CKPT_DIR / ORG_CK), "bound_md5": ORG_CK_MD5,
        "flat_md5": org_flat_md5, "bound_flat_md5": ORG_FLAT_MD5,
        "baseline_panel_g0": baselines,
        "baseline_panel_gm12": {k: float(v)
                                for k, v in baseline_panel12.items()},
        "panel_g0_absdiff_vs_committed": panel_diffs,
        "panel_gm12_absdiff_vs_committed": panel12_diffs,
        "pass": bool(md5of(CKPT_DIR / ORG_CK) == ORG_CK_MD5
                     and org_flat_md5 == ORG_FLAT_MD5
                     and max(panel_diffs.values()) < FACT_READ_TOL_G0
                     and max(panel12_diffs.values()) < FACT_READ_TOL_GM12),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    org_write = org_flat_np - base_flat_np
    metrics["the_organism"] = {
        "checkpoint": f"runs/checkpoints/{ORG_CK}",
        "flat_md5": org_flat_md5,
        "baseline_panel_g0": baselines,
        "write_norm": float(np.linalg.norm(org_write)),
        "in_room_write_norms": [float(np.linalg.norm(
            rooms.project_room(org_write, i))) for i in range(N_ROOMS)],
        "disclosure": REGISTERED["family_disclosure"],
    }
    log("P2 G_FACTLOAD PASS — THE BASELINE TABLE "
        + " ".join(f"{k} {v:.6f}" for k, v in baselines.items())
        + f" | max panel diff {max(panel_diffs.values()):.2e} | flat md5 "
        f"{org_flat_md5} (bit-exact)")
    write_partial("P2 the organism loaded + three-way gated")
    del org_net, org_art

    # ============ P3: THE RECONSTRUCTION (dW_F3 + the family band) ======
    gpu_idle_poll("P3:reconstruction")
    inst_flats = {}
    for f, (ck, md5_lit) in INST_CKS.items():
        art = torch.load(CKPT_DIR / ck, map_location="cpu",
                         weights_only=False)
        assert int(art.get("step", -1)) == 400, f"{ck} not step-400 complete"
        got = md5of(CKPT_DIR / ck)
        assert got == md5_lit, f"{ck} md5 drift: {got} != {md5_lit}"
        inst_flats[f] = flat_params_cpu(G1.evl_load(art["model"])) \
            .double().numpy().astype(np.float64)
        del art
    chain = {"FACT1": base_flat_np}          # the serial chain's bases
    for i in range(1, N_ROOMS):
        chain[FACTS[i]] = inst_flats[FACTS[i - 1]]
    dw_family = {}
    for i, f in enumerate(FACTS):
        dw = inst_flats[f] - chain[f]
        fr = rooms.in_room_fracs(dw)
        dw_family[f] = {
            "dW_l2": float(np.linalg.norm(dw)),
            "in_own_room_norm": float(np.linalg.norm(
                rooms.project_room(dw, i))),
            "in_own_room_frac": fr[i],
            "in_room_frac_per_room": fr,
            "in_union_frac": float(np.sqrt(sum(x * x for x in fr))),
        }
    dw_f3 = inst_flats["FACT3"] - inst_flats["FACT2"]
    dw_f3_norm = float(np.linalg.norm(dw_f3))
    scalpel_vec = rooms.project_room(dw_f3, TARGET_IDX)      # P_room3(dW_F3)
    scalpel_norm = float(np.linalg.norm(scalpel_vec))
    scalpel_frac = scalpel_norm / dw_f3_norm
    own_fr = [dw_family[f]["in_own_room_frac"] for f in FACTS]
    band_lo, band_hi = min(own_fr), max(own_fr)
    recon_checks = {
        "norm": {"mine": dw_f3_norm, "committed_e294": E294_F3WRITE_NORM,
                 "absdiff": abs(dw_f3_norm - E294_F3WRITE_NORM)},
        "in_own_room": {"mine": scalpel_norm,
                        "committed_e294": E294_F3WRITE_IN_OWN_ROOM,
                        "absdiff": abs(scalpel_norm
                                       - E294_F3WRITE_IN_OWN_ROOM)},
        "in_own_frac": {"mine": scalpel_frac,
                        "committed_e294": E294_F3WRITE_IN_OWN_FRAC,
                        "absdiff": abs(scalpel_frac
                                       - E294_F3WRITE_IN_OWN_FRAC)},
    }
    G_RECON = {
        "form": ("THE RECONSTRUCTION (disclosed): dW_FACT3 := flat(F3 "
                 "install end s400) - flat(F2 install end s400) — FACT3's "
                 "own install write on e291's serial chain (base -> F1 -> "
                 "-> F5 -> organism; F1's base := e001); GATED vs e294's "
                 "committed F3WRITE literals at 1e-9; THE FAMILY BAND: "
                 "all five installs' in-own-room shares — FACT3's must "
                 "sit inside the band"),
        "reconstruction_checks": recon_checks,
        "family_band_in_own_room_frac": {
            "per_install": {f: dw_family[f]["in_own_room_frac"]
                            for f in FACTS},
            "band": [band_lo, band_hi],
            "fact3_in_band": bool(band_lo <= scalpel_frac <= band_hi),
            "note": "each install's gradient was projected into ITS OWN "
                    "room at every step (e291's install_fact) — the band "
                    "is the protocol's own leakage residue (AdamW wd + "
                    "fp32 write rounding)"},
        "per_install_dW": dw_family,
        "dw_f3_per_room_spectrum": {
            "form": "dW_F3's in-room fractions across ALL five rooms "
                    "(the 'overlapping but not identical' datum — the "
                    "siblings' rooms' share of FACT3's own write; "
                    "co-report, never a bar)",
            "in_room_frac_per_room": dw_family["FACT3"][
                "in_room_frac_per_room"],
            "in_union_frac": dw_family["FACT3"]["in_union_frac"],
            "union_proj_norm": float(np.linalg.norm(
                rooms.project_union(dw_f3)))},
        "overshoot_disclosure": (
            f"||P_room3(dW_F3)|| {scalpel_norm:.4f} vs the organism's "
            "room3 write norm "
            f"{metrics['the_organism']['in_room_write_norms'][2]:.4f} — "
            "the later installs' AdamW weight decay shrank room3 after "
            "FACT3's install; alpha >= 1.0 subtracts MORE than the "
            "organism's current room3 content (alpha 1.5 flips part of "
            "the spectrum's sign) — the dispatch's own dial, disclosed, "
            "never a bar"),
        "pass": bool(all(v["absdiff"] <= E294_RECON_TOL
                         for v in recon_checks.values())
                     and band_lo <= scalpel_frac <= band_hi),
    }
    assert G_RECON["pass"], f"G_RECON FAILED: {G_RECON}"
    metrics["gates"]["G_RECON"] = G_RECON
    metrics["the_scalpel"] = {
        "dw_f3_norm": dw_f3_norm,
        "scalpel_vec_norm": scalpel_norm,
        "in_own_frac": scalpel_frac,
        "per_alpha_subtracted_norm": {str(a): a * scalpel_norm
                                      for a in sweep},
        "transport_bracket_reference": (
            "e290's realized-drift bracket "
            f"[{E290_DRIFT_BRACKET[0]:.8f}, {E290_DRIFT_BRACKET[1]:.8f}] "
            f"x ||F3WRITE|| = [{E290_DRIFT_BRACKET[0] * dw_f3_norm:.6f}, "
            f"{E290_DRIFT_BRACKET[1] * dw_f3_norm:.6f}] — alpha=0.25's "
            f"subtraction ({0.25 * scalpel_norm:.4f}) already sits "
            "~230x above the passive kill bracket: THE SCALPEL IS A "
            "BEARER REMOVAL, NOT A TRANSPORT PERTURBATION"),
    }
    log(f"P3 G_RECON PASS — dW_F3 {dw_f3_norm:.6f} (committed "
        f"{E294_F3WRITE_NORM:.6f}, diff "
        f"{abs(dw_f3_norm - E294_F3WRITE_NORM):.1e}); in-own-room "
        f"{scalpel_norm:.6f} frac {scalpel_frac:.4f}; family band "
        f"[{band_lo:.4f}, {band_hi:.4f}] — FACT3 in band; dW_F3's sibling-"
        "room fracs "
        + " ".join(f"{FACTS[j]}:{dw_family['FACT3']['in_room_frac_per_room'][j]:.4f}"
                   for j in range(N_ROOMS) if j != TARGET_IDX))
    del inst_flats, chain
    write_partial("P3 the reconstruction gated (dW_F3 vs e294's literals "
                  "+ the family band)")

    # ============ P4: THE PROJECTOR IDENTITY (e310's G_ORTH form) ======
    prng = np.random.default_rng(E261.CERT_SEED)
    idem_list, cross_list, energy_list = [], [], []
    probes_id = [dw_f3] + [prng.standard_normal(N) for _ in range(3)]
    for x in probes_id:
        px = rooms.project_room(x, TARGET_IDX)
        cx = x - px
        ppx = rooms.project_room(px, TARGET_IDX)
        nx, npx, ncx = (np.linalg.norm(x), np.linalg.norm(px),
                        np.linalg.norm(cx))
        idem_list.append(float(np.linalg.norm(ppx - px) / max(npx, 1e-30)))
        cross_list.append(float(abs(px @ cx)
                                / max(npx * ncx, 1e-30)))
        energy_list.append(float(abs(npx ** 2 + ncx ** 2 - nx ** 2)
                                 / max(nx ** 2, 1e-30)))
    # the scalpel vec's own in-room residual (P(P dW) == P dW)
    resid_scalpel = float(np.linalg.norm(
        rooms.project_room(scalpel_vec, TARGET_IDX) - scalpel_vec)
        / scalpel_norm)
    G_PROJID = {
        "form": ("THE PROJECTOR IDENTITY on dW_F3 + 3 fresh probes "
                 "(e310's G_ORTH convention): idempotency, complement "
                 "orthogonality, energy conservation; bars 1e-9"),
        "idempotency_max": max(idem_list),
        "cross_dot_rel_max": max(cross_list),
        "energy_sum_rel_resid_max": max(energy_list),
        "scalpel_vec_in_room_resid": resid_scalpel,
        "bars": {"idempotency": PROJ_ID_BAR, "cross": PROJ_ID_BAR,
                 "energy": PROJ_ID_BAR},
        "pass": bool(max(idem_list) <= PROJ_ID_BAR
                     and max(cross_list) <= PROJ_ID_BAR
                     and max(energy_list) <= PROJ_ID_BAR),
    }
    assert G_PROJID["pass"], f"projector identity FAILED: {G_PROJID}"
    metrics["gates"]["G_PROJID"] = G_PROJID
    log(f"P4 G_PROJID PASS — idem {max(idem_list):.1e} / cross "
        f"{max(cross_list):.1e} / energy {max(energy_list):.1e} / the "
        f"scalpel vec's in-room resid {resid_scalpel:.1e}")
    write_partial("P4 the projector identity PASSED")

    # ============ P5: THE SHAM (constructed + gated) ====================
    shrng = np.random.default_rng(SHAM_SEED)
    r_raw = shrng.standard_normal(N)
    sham_p = rooms.project_room(r_raw, TARGET_IDX)
    sham_p_norm_raw = float(np.linalg.norm(sham_p))
    sham_vec = sham_p * (scalpel_norm / sham_p_norm_raw)
    sham_inroom_resid = float(np.linalg.norm(
        rooms.project_room(sham_vec, TARGET_IDX) - sham_vec)
        / scalpel_norm)
    cos_sham_to_scalpel = float(sham_vec @ scalpel_vec
                                / (scalpel_norm * scalpel_norm))
    G_SHAM = {
        "form": ("THE SHAM CONTROL: ONE fresh registered direction (seed "
                 f"{SHAM_SEED}, the family's per-cell rule): r ~ N(0,1)^N "
                 "fp64 -> P_room3(r) -> norm-matched EXACTLY to "
                 "||P_room3(dW_F3)||; the same alphas, the same probes — "
                 "'does ANY in-room displacement of this size kill FACT3, "
                 "or only ITS OWN tail?'"),
        "seed": SHAM_SEED,
        "raw_proj_norm": sham_p_norm_raw,
        "norm_matched_to": scalpel_norm,
        "norm_rel_err": abs(float(np.linalg.norm(sham_vec)) - scalpel_norm)
        / scalpel_norm,
        "in_room_resid": sham_inroom_resid,
        "cosine_to_scalpel": cos_sham_to_scalpel,
        "chance_cosine_ref": math.sqrt(ROOM_K / N),
        "pass": bool(abs(float(np.linalg.norm(sham_vec)) - scalpel_norm)
                     / scalpel_norm <= SHAM_NORM_TOL
                     and sham_inroom_resid <= PROJ_ID_BAR),
    }
    assert G_SHAM["pass"], f"G_SHAM FAILED: {G_SHAM}"
    metrics["gates"]["G_SHAM"] = G_SHAM
    log(f"P5 G_SHAM PASS — in-room random direction, norm matched to "
        f"{scalpel_norm:.6f} (rel err "
        f"{G_SHAM['norm_rel_err']:.1e}); cos-to-scalpel "
        f"{cos_sham_to_scalpel:+.4f} (chance ~+-{G_SHAM['chance_cosine_ref']:.2f})")
    write_partial("P5 the sham constructed + gated")

    # ---- the vectors artifact (gitignored; md5s in metrics) -------------
    vec_path = RD / ("e314_smoke_vectors.pt" if SMOKE
                     else "e314_tail_vectors.pt")
    torch.save({"scalpel_vec_fp64": torch.from_numpy(scalpel_vec),
                "sham_vec_fp64": torch.from_numpy(sham_vec),
                "meta": {"experiment": NAME, "seed_sham": SHAM_SEED,
                         "organism_flat_md5": org_flat_md5,
                         "dw_f3_norm": dw_f3_norm,
                         "scalpel_norm": scalpel_norm}}, vec_path)
    metrics["vectors_artifact"] = {
        "path": str(vec_path.relative_to(E43.REPO)),
        "scalpel_vec_md5": hashlib.md5(
            scalpel_vec.tobytes()).hexdigest(),
        "sham_vec_md5": hashlib.md5(sham_vec.tobytes()).hexdigest(),
        "note": "the EXACT fp64 subtracted vectors; theta_alpha := "
                "organism_fp64 - alpha * vec, cast fp32 — the applied "
                "states' flat md5s in the sweep tables",
    }

    # ============ P6: THE SWEEP (both arms, one panel per alpha) ========
    evl = G1.evl_load(organism_sd)
    evl.eval()

    def panel_at(theta64: np.ndarray, tag: str) -> dict:
        theta32 = torch.from_numpy(theta64.astype(np.float32))
        apply_flat(evl, theta32, rooms.offsets, rooms.shapes)
        g0 = {f: float(G1.battery_cell(evl, g0_ids_f[f], zid)["mean_pz"])
              for f in FACTS}
        gm = {f: float(G1.battery_cell(evl, gm12_ids_f[f], zid)["mean_pz"])
              for f in FACTS}
        fmd5 = hashlib.md5(theta32.numpy().tobytes()).hexdigest()
        log(f"  [{tag}] panel g0 "
            + " ".join(f"{f}:{g0[f]:.6f}" for f in FACTS)
            + f" | flat md5 {fmd5[:10]}")
        return {"g0": g0, "gm12": gm, "flat_md5": fmd5}

    sweep_tables = {}
    for arm, vec in (("SCALPEL", scalpel_vec), ("SHAM", sham_vec)):
        gpu_idle_poll(f"{arm}:sweep")
        rows = {}
        for alpha in sweep:
            theta_a = org_flat_np - alpha * vec
            rows[str(alpha)] = panel_at(theta_a, f"{arm}:a={alpha}")
            sweep_tables.setdefault(arm, {})[str(alpha)] = rows[str(alpha)]
            metrics["sweep"] = {
                "form": ("the five-fact g0 panel (the bar) + gm12 "
                         "(co-report, never a bar) at every alpha, both "
                         "arms; alpha=0 := the organism's own baseline "
                         "panel (G_FACTLOAD)"),
                "baseline_alpha0": {"g0": baselines,
                                    "gm12": {k: float(v) for k, v in
                                             baseline_panel12.items()}},
                "tables": sweep_tables,
            }
            write_partial(f"P6 {arm} alpha={alpha} probed")
        del rows

    # ---- the applied displacements' ledger --------------------------------
    metrics["displacement_ledger"] = {
        f"{arm}:a={a}": {"subtracted_norm": a * scalpel_norm,
                         "as_frac_of_organism_write":
                             a * scalpel_norm
                             / metrics["the_organism"]["write_norm"],
                         "as_frac_of_transport_bracket_hi":
                             a * scalpel_norm
                             / (E290_DRIFT_BRACKET[1] * dw_f3_norm)}
        for arm in ARMS for a in sweep}

    # ============ P7: THE ADJUDICATION (the frozen bars) =================
    def kill_set_of(arm: str) -> list:
        return [a for a in sweep
                if sweep_tables[arm][str(a)]["g0"][TARGET_FACT] < ERASE_BAR]

    def hold_count(arm: str, alpha) -> int:
        row = sweep_tables[arm][str(alpha)]["g0"]
        return sum(1 for f in FACTS if f != TARGET_FACT
                   and row[f] >= HOLD_FRAC * baselines[f])

    ks = kill_set_of("SCALPEL")
    ks_sham = kill_set_of("SHAM")
    if ks:
        alpha_kill = ks[0]
        hc = hold_count("SCALPEL", alpha_kill)
        sham_kills = bool(sweep_tables["SHAM"][str(alpha_kill)]["g0"]
                          [TARGET_FACT] < ERASE_BAR)
        if hc >= SIBLING_MIN_HOLD and not sham_kills:
            word = "SURGICAL-GEOMETRY"
        elif hc <= 2:
            word = "EVERYTHING-IS-SHARED"
        else:
            word = "MIXED"
        best_alpha = alpha_kill
    else:
        alpha_kill = None
        hc = hold_count("SCALPEL", sweep[-1])
        sham_kills = None
        word = "NOTHING-DIES"
        best_alpha = min(sweep, key=lambda a: sweep_tables["SCALPEL"]
                         [str(a)]["g0"][TARGET_FACT])
    clause = {
        "SURGICAL-GEOMETRY": (
            f"alpha={alpha_kill} kills FACT3 "
            f"({sweep_tables['SCALPEL'][str(alpha_kill)]['g0'][TARGET_FACT]:.2e} "
            f"< {ERASE_BAR}) with {hc}/4 siblings >= {HOLD_FRAC:.0%}x "
            f"baseline AND the sham does NOT kill at the same alpha "
            f"(sham FACT3 {sweep_tables['SHAM'][str(alpha_kill)]['g0'][TARGET_FACT]:.2e}) "
            "— the first selective unlearning, bought at the geometry"),
        "EVERYTHING-IS-SHARED": (
            f"alpha={alpha_kill} kills FACT3 but only {hc}/4 siblings "
            f"hold >= {HOLD_FRAC:.0%}x — the siblings fall with it; the "
            "indivisibility is IN THE TAILS"),
        "NOTHING-DIES": (
            "no alpha kills FACT3 even at 1.5x — best (lowest) read "
            f"{sweep_tables['SCALPEL'][str(best_alpha)]['g0'][TARGET_FACT]:.4f} "
            f"at alpha={best_alpha}; the read's bearer is not FACT3's "
            "own install tail — the bearer law's family reading needs "
            "revision"),
        "MIXED": (
            f"alpha={alpha_kill} kills FACT3 with {hc}/4 siblings "
            "holding; sham_kills_at_same_alpha="
            f"{sham_kills} — the honest residual reading; the alpha "
            "table verbatim (the dose-response IS the deliverable)"),
    }[word]
    adjudication = {
        "word": word,
        "clause": clause,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "operationalization": (
            "kill_set := {alpha : scalpel g0 FACT3 < 0.01}; alpha_kill := "
            "min(kill_set) (the dose-response's first kill); "
            "hold_count(alpha) := #{siblings >= 0.5x the loaded "
            "organism's own committed baseline}; the sham clause read at "
            "THE SAME alpha; NOTHING-DIES iff kill_set empty; else "
            "SURGICAL-GEOMETRY iff hold >= 3 AND sham no-kill; else "
            "EVERYTHING-IS-SHARED iff hold <= 2; else MIXED"),
        "alpha_kill": alpha_kill,
        "best_alpha_if_no_kill": None if ks else best_alpha,
        "hold_count_at_alpha_kill": hc if ks else None,
        "sham_kills_at_alpha_kill": sham_kills,
        "scalpel_kill_set": [str(a) for a in ks],
        "sham_kill_set": [str(a) for a in ks_sham],
        "hold_counts_per_alpha": {
            arm: {str(a): hold_count(arm, a) for a in sweep}
            for arm in ARMS},
        "sibling_holds_detail": {
            arm: {str(a): {f: {"read": sweep_tables[arm][str(a)]["g0"][f],
                               "baseline": baselines[f],
                               "ratio": (sweep_tables[arm][str(a)]["g0"][f]
                                         / baselines[f]),
                               "holds": bool(
                                   sweep_tables[arm][str(a)]["g0"][f]
                                   >= HOLD_FRAC * baselines[f])}
                          for f in FACTS if f != TARGET_FACT}
                  for a in sweep}
            for arm in ARMS},
        "smoke_stamp": bool(SMOKE),
    }
    if SMOKE:
        adjudication["word"] = "SMOKE-NOT-ADJUDICATED"
        adjudication["clause"] = ("SMOKE: the pipeline's gates + one "
                                  "alpha probed; NOTHING adjudicated")
    metrics["adjudication"] = adjudication
    log("=" * 78)
    log(f"P7 THE VERDICT ({'SMOKE' if SMOKE else 'ADJUDICATED'}): "
        f"{adjudication['word']} — {clause}")
    write_partial("P7 adjudicated" if not SMOKE else "P7 smoke complete")

    # ============ P8: THE FIGURE =========================================
    png = RD / ("e314_smoke_tail_scalpel.png" if SMOKE
                else "e314_tail_scalpel.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    colors = {f: plt.get_cmap("viridis")(i / (N_ROOMS - 1))
              for i, f in enumerate(FACTS)}
    for ax, arm in zip(axes, ARMS):
        xs = [0.0] + [float(a) for a in sweep]
        for f in FACTS:
            ys = [baselines[f]] + [sweep_tables[arm][str(a)]["g0"][f]
                                   for a in sweep]
            lw, z = (3.0, 5) if f == TARGET_FACT else (1.6, 3)
            ax.plot(xs, ys, "-o", color=colors[f], lw=lw, zorder=z,
                    markersize=5,
                    label=f + (" (TARGET)" if f == TARGET_FACT else ""))
            ax.axhline(HOLD_FRAC * baselines[f], color=colors[f],
                       ls=":", lw=0.8, alpha=0.5)
        ax.axhline(ERASE_BAR, color="red", ls="--", lw=1.2, alpha=0.9)
        ax.text(0.02, ERASE_BAR + 0.004, "kill bar 0.01", color="red",
                fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_xlabel("alpha (fraction of P_room3(dW_F3) subtracted)")
        ax.set_title({"SCALPEL": "(a) THE SCALPEL — theta - a*P_room3(dW_F3)",
                      "SHAM": "(b) THE SHAM — same-norm random in-room"}[arm],
                     fontsize=10)
        ax.set_xticks(xs)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("five-fact g0 panel read (mean p(Z))")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("E314 THE TAIL SCALPEL — the geometric erase on the "
                 "five-family organism | verdict: "
                 + str(adjudication["word"]), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=140)
    plt.close(fig)
    metrics["outputs"] = {"figure": str(png.relative_to(E43.REPO)),
                          "metrics": str(METRICS_PATH.relative_to(E43.REPO)),
                          "report": None}
    log(f"P8: figure saved {png.name}")
    write_partial("P8 figure saved")

    # ============ P9: THE REPORT (executor-written) ======================
    if not SMOKE:
        rep = RD / "REPORT.md"
        lines = ["# E314 — THE TAIL SCALPEL (the geometric erase)",
                 "",
                 f"**Verdict: {word}** — {clause}",
                 "",
                 "## The alpha table (both arms; the deliverable)",
                 "",
                 "| arm | alpha | FACT1 | FACT2 | FACT3 (T) | FACT4 | "
                 "FACT5 | holds (>=0.5x) |",
                 "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            for a in sweep:
                row = sweep_tables[arm][str(a)]["g0"]
                lines.append(
                    f"| {arm} | {a} | "
                    + " | ".join(f"{row[f]:.4f}" for f in FACTS)
                    + f" | {hold_count(arm, a)}/4 |")
            lines.append(f"| {arm} | 0 (base) | "
                         + " | ".join(f"{baselines[f]:.4f}"
                                      for f in FACTS)
                         + " | 4/4 |")
        lines += [
            "",
            "## The reconstruction (disclosed)",
            "",
            "- dW_FACT3 := flat(e291_install_F3_resume s400) - "
            "flat(e291_install_F2_resume s400) — FACT3's own "
            "install write on e291's serial chain (F1's base = "
            "e001); gated vs e294's committed F3WRITE literals at "
            "1e-9 (all three reproduced).",
            f"- ||dW_F3|| = {dw_f3_norm:.6f}; "
            f"||P_room3(dW_F3)|| = {scalpel_norm:.6f} "
            f"({scalpel_frac:.2%} of the write); the family band "
            f"of in-own-room shares [{band_lo:.4f}, {band_hi:.4f}].",
            "- OVERSHOOT DISCLOSURE: the organism's current room3 "
            "write norm is "
            f"{metrics['the_organism']['in_room_write_norms'][2]:.4f} "
            f"< {scalpel_norm:.4f} — the later installs' AdamW "
            "weight decay shrank room3 after FACT3's install; "
            "alpha >= 1.0 subtracts more than the room holds "
            "(alpha 1.5 sign-flips part of the spectrum).",
            "",
            "## The sham",
            "",
            f"- ONE registered direction (seed {SHAM_SEED}): "
            f"P_room3(N(0,1)^N), norm-matched to {scalpel_norm:.6f} "
            f"(rel err {G_SHAM['norm_rel_err']:.1e}); "
            f"cos-to-scalpel {cos_sham_to_scalpel:+.4f}.",
            f"- sham kill set: {[str(a) for a in ks_sham] or 'none'}.",
            "",
            "## The three roads, closed or open",
            "",
            "- e312 (composition): MIXED — the protectors "
            "resurrect the target.",
            "- e313 (sequencing): SEQ-LEAKY — the membrane "
            "remembers past death.",
            f"- e314 (the geometry): **{word}**.",
            "",
            "Registered predictions at birth: "
            + "; ".join(f"{k}: {v[:150]}"
                        for k, v in
                        REGISTERED["predictions"].items()),
            "",
            "Full tables, gates, and provenance: metrics.json "
            "(this directory).",
            ""]
        rep.write_text("\n".join(lines), encoding="utf-8")
        metrics["outputs"]["report"] = str(rep.relative_to(E43.REPO))
        log(f"P9: report written {rep.name}")

    metrics["status"] = ("COMPLETE (smoke; nothing adjudicated)" if SMOKE
                         else "COMPLETE — adjudicated")
    metrics["date_updated"] = common.now_iso()
    metrics["git_head_final"] = git_head()
    gpu_idle_poll("P9:done")
    save_json(METRICS_PATH, metrics)
    log(f"E314 DONE ({'smoke' if SMOKE else 'full'}) — verdict "
        f"{adjudication['word']}")


if __name__ == "__main__":
    main()
