"""X26 — THE FORGED AUTHOR (R73 ideator card 2; now T292's CAUSAL PROBE for
Law 3's mechanism noun; dispatched 2026-10-09). This docstring carries the
question, the background, the design, the bars and P-x26a VERBATIM from the
dispatch letter, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (verbatim): "x26 — THE FORGED AUTHOR (R73 ideator card 2; now
T292's CAUSAL PROBE for Law 3's mechanism noun)."

BACKGROUND (verbatim): "x24 found only gradient-written names lift under
displacement; x25 just proved the lift is NOT the write's mass (it survives
and doubles under the scalpel) — but CAUSALITY is unresolved: does WRITING
cause the fragility, or do writes land on pre-disposable slots (SELECTION)?
THE FORGED AUTHOR is the minimal causal intervention: ascend a robust
never-written name's initial-character row in the lm_head ONLY — 384
parameters, zero context write, zero room write — then re-run the
displacement panel. If forging lifts the slot, authorship is LOCAL
ROW-GRADIENT HISTORY (causality in its minimal form, and era-1's token-row
doctrine reconnects through the fragility where x22 denied it in the knob).
If the forge is inert, authorship requires the full contextual write — and
e330 (the writing test) splits full-causality from selection."

THE DESIGN (verbatim):
  "1. THE TARGET: QELVARO — x24's most robust never-written name (prior
  0.0026, lift 0.50x, the 23.7x-robust datum). Ascend p(Q) via ONLY its
  initial-char 'Q' lm_head row (+ the name's other rows co-reported but
  never the primary arm) to the marginal-prior regime x24's written names
  occupied (target p(Q) ~0.002-0.005 at the host contexts — a LIFT to
  visibility, not a read; norm the row's change and disclose).
  2. THE PANEL (x25's rig verbatim, md5-bound): the complement displacement
  + the gaussian rider on the forged state, reading QELVARO, TAVIREN,
  ZEPHYRA, VIRETAN at host + neutral — the full {complement, gaussian} x
  {forged, base} x 4 names x 2 sites table, with the base columns
  reproducing x24/x25's committed values bit-exact.
  3. Controls: a random-direction row perturbation of the same row at
  matched norm (is it the ascent's alignment or any row touch?); the
  untouched rows' names as the flat band."

BARS (frozen VERBATIM from the dispatch, BEFORE any compute):
  - FORGE-LIFTS: "the forged QELVARO's complement-lift at host rises >= 3x
    over its committed 0.50x band (i.e. >= 1.5x) while the untouched names
    stay in their bands — AUTHORSHIP IS LOCAL ROW-GRADIENT HISTORY;
    causality lands in its minimal form; era-1's doctrine reconnects via
    the fragility; Law 3's noun finalizes as WRITING-CAUSED (row-local
    component)."
  - FORGE-INERT: "the forged QELVARO stays in the never-written band (<= 2x
    of 0.50x) despite its raised prior — authorship is constitutively
    CONTEXTUAL (the write at contexts, not the row); e330 becomes the
    decisive cell; the selection hypothesis strengthens."

REGISTERED PREDICTION (frozen VERBATIM, BEFORE any compute):
  "Register P-x26a BEFORE compute. Lab guess — the ledger's lesson applied,
  REGISTER THE COUNTER-LEAN: after nine straight mechanism-miss losses the
  lab declines a confident guess; the honest prior from x22 (the row
  delivered 0.8% of the knob) leans INERT, and x25's doubling (removing
  mass raised the lift) argues the slot's disposition is NOT in the head —
  state your own read if you differ; predictions are scored."

  EXECUTOR READ (stated under the dispatch's invitation, scored):
  - P-x26a-exec: INERT — the executor does not differ on the word; own
    reasoning registered pre-compute: (i) x25's rider showed the lifted
    slots couple to GENERIC energy (TAVIREN's gaussian/complement ratios
    0.149 host / 0.680 neutral) — a 192-coordinate OUTPUT-row change cannot
    re-create a trunk-side bearer disposition; (ii) the row sits after
    ln_f, downstream of the residual bearer x19 measured; (iii) x22's knob
    reading (the row delivered 0.8% of the knob). Scores TRUE iff the
    verdict == FORGE-INERT.

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ====

* THE PRIMARY CURRENCY (the currency catch, disclosed + frozen): x24's raw
  lift currency p(displaced)/p_base would CONFLATE the forge's own prior
  bump (~1.9x by construction) with the displacement's effect — under it
  the INERT bar (<= 2x of 0.50x) would be unreachable and the LIFTS bar
  trivially fireable, so the two bars are only coherent as a fork under the
  same-organism DISPLACEMENT-LIFT: displift(core, disp, name, site) :=
  p(core+disp, name, site) / max(p(core, name, site), 1e-12), core in
  {base, forged, rnd} (x25's displift rider currency; at core=base it IS
  x24's committed lift). The raw x24-currency lift is co-reported for
  every cell, never adjudicated.
* THE FORGE (the e043-class ascent convention ported to ONE row): organism
  := e001 (md5-bound); target row := lm_head.weight[stoi['Q']] (cid 29,
  read from the md5-bound x24 metrics, never retyped); objective := mean
  over the 60 host-g0 contexts of -log p(Q at the final position) — the
  panel's own read, the most favorable case for FORGE-LIFTS; optimizer :=
  AdamW(betas=(0.9,0.95), wd 0.1), lr 1e-3, house cosine_lr(step-1,
  total=400, warmup=100), grad-clip 1.0 — e043's exposure convention
  (full-batch, the deterministic limit; no sampling RNG); MAX_STEPS 400.
* THE TARGET REGIME (the target-band catch, disclosed + frozen): x24's
  committed QELVARO prior at host (0.0025906) ALREADY sits inside the
  dispatch's [0.002, 0.005] regime, so "ascend to the regime" is
  operationalized as ascend to the TOP of the band: early-stop at the
  first step whose mean p(Q) at host >= 0.0050 (a ~1.9x prior lift — "a
  LIFT to visibility, not a read"); accepted landing band [0.0040, 0.0120]
  (gate); any single step above 0.02 halts to TEXTURE (the forge became a
  read — regime change). The row's change is normed and disclosed (the
  ROW-NORM LADDER: per-step row delta L2 + p(Q) trajectory, in metrics +
  figure).
* THE 384-CATCH (disclosed + frozen): the dispatch's "384 parameters" is
  corrected by the organism's own cfg — the 2.74M g1b family is n_embd=192,
  so ONE lm_head row is 192 coordinates; the design's operative constraint
  (its own gate) is "ONLY the target row's coordinates move", verified
  bitwise (exactly 192 changed elements, all in row cid, every other key
  bit-identical). The wte row is NOT touched ("in the lm_head ONLY").
* THE RANDOM-ROW CONTROL (the design's control, frozen): one fresh draw
  (seed 26001, this cell's ONLY fresh random vector): a standard-normal
  direction in R^192, L2-normalized to the forge's fp32 row-delta norm,
  added to the same row of the base organism. Gated: fp32 norm match at
  1e-6 relative; same bitwise row-accounting. The control clause
  (co-report, never a dispatch bar): RANDOM-TOUCH-GENERIC iff the control's
  own displift(rnd, comp, QELVARO, host) also >= the 1.5x bar (the lift is
  row-touch-generic, not ascent-aligned); else ALIGNED-ONLY.
* THE PANEL (frozen, x25's rig verbatim): 4 names — QELVARO (the forged
  target; never-written fresh-bank), TAVIREN (written; the sole clean x24
  datum), ZEPHYRA (written anchor; the x24-disclosed confound carried),
  VIRETAN (never-written scrambled control); 2 sites — HOST-G0 (e261's
  splice bank, mix FLORIZEL 19 / ELIZABETH 41, [60,130]) + NEUTRAL (x24's
  frozen rule, seed 24001, n_candidates gated == x24's committed count).
  States (one applied net each, probed at both sites on all 4 names):
  base; live_comp = the x15 carrier's own model (x24's exact state);
  live_gauss = base + gaussian; forged; forged_comp; forged_gauss; rnd;
  rnd_comp; rnd_gauss.
* THE DISPLACEMENTS: (i) THE COMPLEMENT — the K10K write's complement at
  full dose, x15's construction rebuilt bit-exact AND verified bit-equal to
  the committed carrier runs/x15/x15_comp_xFULL.pt (G_ORTH + G_DOSEMATCH
  1e-12 + G_INROOM 1e-12 + G_REGEN every key); injected as state_fp32 +
  carrier_delta_fp32 (x15's own method). (ii) THE GAUSSIAN — x25's OWN
  draw (seed 25001), rebuilt bit-identically, NOT fresh: the dispatch's
  own gate demands the base column reproduce x25's committed values
  bit-exact, which pins the draw to x25's (md5-lineage-bound thereby);
  norm 9.1788432658723, in-room fractions at chance (x25's bands).
* THE UNTOUCHED BANDS (the LIFTS bar's "untouched names stay in their
  bands" clause, frozen): for each of TAVIREN / ZEPHYRA / VIRETAN,
  displift(forged, comp, ., host) within [2/3, 1.5] x its committed x24
  host lift {11.8422, 54.6027, 0.4408} (read from the md5-bound artifact,
  cross-checked vs frozen literals). If any untouched name leaves its
  band, attribution is broken -> MIXED regardless of the numeric clause.
* VERDICT WORDS (frozen): the verdict in {FORGE-LIFTS, FORGE-INERT, GAP,
  MIXED} on Q_h := displift(forged, comp, QELVARO, host_g0): (1) any
  untouched name out of band -> MIXED; (2) FORGE-LIFTS iff Q_h >= 3 x
  committed (1.4977860725783164, read from the md5-bound x24 artifact);
  (3) FORGE-INERT iff Q_h <= 2 x committed (0.9985240483855442); (4) the
  window between -> GAP (intermediate; e330 decisive; no noun change
  without a new registered cell). The prediction scores against the
  registering side (P-x26a lab vs P-x26a-exec, both INERT-lean words).
* CE_R SCOPING (frozen): the [1.0, 2.2] sanity band applies to the
  UNDISPLACED states (base, forged, rnd) — a forged organism that torches
  the corpus read flags the kill-everything confound; displaced states'
  ce_r are HONESTY RIDERS (co-reported, never bars).
* GATES (a failure HALTS => TEXTURE, nothing adjudicated): G_ENDPOINTS
  (9 md5 binds: e001, e261_K10K_inst_resume, e264_rooms, x15 metrics +
  carrier, x24 metrics, x25 metrics, e293 bank source, e043 ascent source;
  + committed references READ FROM THE ARTIFACTS and cross-checked against
  frozen literals, never retyped), G_FLATBASIS, G_PANEL, G_NAMEFREE,
  G_BATTERY, G_NEUTRAL (x24's n_candidates), G_ROOM (D/S bit-equal),
  G_ORTH, G_DOSEMATCH, G_INROOM, G_REGEN (the complement bit-equal to its
  carrier), G_GAUSS (x25's draw rebuilt: norm + chance fractions),
  G_FORGE (the ascent: target regime hit, trajectory recorded, convention
  frozen), G_FORGEACCOUNT (bitwise: ONLY the target row's 192 coordinates
  move — forged AND rnd), G_ROWMATCH (the control's fp32 norm matched),
  G_FORGEREAD (the ascent's final measured p(Q) vs the site-read p(Q) at
  1e-6), G_BASEPRIOR, G_X24REPRO (live_comp vs x24 committed, |d| <=
  1e-12), G_X25REPRO (live_gauss vs x25 committed, |d| <= 1e-12),
  G_HOSTSANITY (ce_r [1.0, 2.2] on base/forged/rnd), G_ONESTATE.
* REGISTERED SEEDS (frozen): global 26000; random-row control 26001 (the
  ONLY fresh draw); neutral bank 24001 (x24's own — site identity); room
  cert 24005; gaussian 25001 (x25's own, inherited for bit-repro).
* THE OTHER ROWS (the design's co-report arm): QELVARO's other six chars'
  host probabilities + their displifts under the forged state co-reported
  from the same site passes; never a separate ascent arm.
* Outputs: runs/x26/{metrics.json (PROGRESSIVE), REPORT.md,
  x26_forged_author.png} (the panel bars: forged vs base vs control per
  name per displacement, both sites; the row-norm ladder).
* Smoke (X26_SMOKE=1): the FULL gate path + the FULL ascent + ALL reads
  live at every state; own smoke dir runs/x26_smoke/; NO adjudication, NO
  figure, NO report (SMOKE stamp on every read; nothing adjudicated).

COMPUTE: CPU-ONLY (the dispatch's own feasibility ruling: the row is 192
trainable coordinates; full-batch 60x130 gradient steps run in seconds).
~30 ascent steps + 18 site-state forward passes + 9 ce_r evals + 2 DCT
projections on the 2.74M organism — minutes, CPU-only.

Run:  python lab/x26_forged_author.py          (X26_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""         # CPU-ONLY, forced

import hashlib                                   # noqa: E402
import json                                      # noqa: E402
import math                                      # noqa: E402
import random                                    # noqa: E402
import re as _re                                 # noqa: E402
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
from common import (CharCorpus, cosine_lr,       # noqa: E402
                    run_dir, save_json, set_seed)

import e043_install as E43                       # noqa: E402 (REPO,
                                                  # find_occ, SPLICE_RNG)
import g1b_continuity as GB                      # noqa: E402 — MUST be
                                                  # imported BEFORE G1
import g1_anchored_ball as G1                    # noqa: E402
import e261_rank_ladder as E261                  # noqa: E402
from e261_rank_ladder import SRCT                # noqa: E402

torch.set_num_threads(4)          # CPU-only cell; the shared desk lane
E261.DCT_WORKERS = 4              # x15/x17/x23/x24/x25's desk convention

SMOKE = os.environ.get("X26_SMOKE") == "1"
NAME = "x26_smoke" if SMOKE else "x26"

T0 = time.time()
RD = run_dir(NAME)
REPO = E43.REPO
CKPT_DIR = GB.CKPT_DIR


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(m: str) -> None:
    print(f"[x26 {time.time() - T0:7.1f}s] {m}", flush=True)


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
N_EMBD = 192                                      # g1b cfg (the 384-catch)
GLOBAL_SEED = 26000
ROWRND_SEED = 26001                               # the ONLY fresh draw
NEUTRAL_SEED = 24001                              # x24's own (site identity)
CERT_SEED = 24005
GAUSS_SEED = 25001                                # x25's own (inherited)

# ---- the parents' committed artifacts (md5-bound) --------------------------
X15_METRICS = REPO / "runs" / "x15" / "metrics.json"
X15_METRICS_MD5 = "c90dd9371a5a7e248fdbd92c06bb243a"
X15_COMP_CARRIER = REPO / "runs" / "x15" / "x15_comp_xFULL.pt"
X15_COMP_CARRIER_MD5 = "b0a1785c6f759fdd1d8deeae7fee4a9b"
X24_METRICS = REPO / "runs" / "x24" / "metrics.json"
X24_METRICS_MD5 = "7f1a91cc1d42a202c8fb23c4f8fb3e7e"
X25_METRICS = REPO / "runs" / "x25" / "metrics.json"
X25_METRICS_MD5 = "aa711be2785ce33281796fc3fc5653a0"
E293_SOURCE = REPO / "lab" / "e293_distinct_contention.py"   # the bank
E293_SOURCE_MD5 = "1f3e16968359171d3db1a52824c48cf4"
E043_SOURCE = REPO / "lab" / "e043_install.py"   # the ascent convention
E043_SOURCE_MD5 = "f82806b369452b05b8d89ca6cebe70fa"

# ---- frozen literals (cross-checked against the md5-bound artifacts) ------
K10K_WRITE_NORM = 9.1788432658723                 # ||dW_k|| (x15/e318)
X15_S_FULL = 3.0349885643664365                   # x15's committed scalar
X15_COMP_L2_64 = 3.024341960836486                # natural complement L2
GAUSS_NORM = 9.1788432658723                      # x25's matched dose
GAUSS_INROOM_NORM_BAND = (0.055, 0.066)           # chance sqrt(k/N)=0.060423
GAUSS_INROOM_ENERGY_BAND = (0.0034, 0.0039)       # chance k/N=0.0036509
E293_BANK = ("ZEPHYRA", "TAVIREN", "QELVARO", "BUVONDI", "NYSTORA")

# ---- the forge (frozen; e043's convention ported to ONE row) ---------------
FORGE_LR = 1e-3
FORGE_WD = 0.1
FORGE_BETAS = (0.9, 0.95)
FORGE_CLIP = 1.0
FORGE_WARMUP = 100                                # the house default
FORGE_MAX_STEPS = 400
FORGE_STOP_P = 0.0050                             # the top of [0.002, 0.005]
FORGE_BAND = (0.0040, 0.0120)                     # accepted landing regime
FORGE_HALT_P = 0.02                               # above -> TEXTURE
FORGEREAD_TOL = 1e-6                              # ascent vs site-read

# ---- the bars (frozen; committed refs READ from artifacts, cross-checked) --
X24_QELVARO_LIFT_HOST = 0.4992620241927721        # the committed 0.50x band
X24_QELVARO_PRIOR_HOST = 0.0025905624497681856
X24_UNTOUCHED_LIFTS_HOST = {
    "TAVIREN": 11.84216068203347,
    "ZEPHYRA": 54.60267494437959,
    "VIRETAN": 0.4408284124472304,
}
FORGE_LIFTS_MULT = 3.0                            # ">= 3x over its band"
FORGE_INERT_MULT = 2.0                            # "<= 2x of 0.50x"
UNTOUCHED_BAND = (2.0 / 3.0, 1.5)
REPRO_TOL = 1e-12                                 # "bit-exact"
CE_BAND = (1.0, 2.2)                              # e318's sanity band
PRIOR_FLOOR = 1e-12
BASE_BAR = 0.05                                   # e311's G_BASE bar
ROWMATCH_TOL = 1e-6                               # relative fp32 norm match

# ---- the panel (frozen) ----------------------------------------------------
PANEL = [
    # (name, class, role)
    ("QELVARO",  "synthetic_count0_fresh_bank", "forged_target_never_written"),
    ("TAVIREN",  "synthetic_count0",            "written"),
    ("ZEPHYRA",  "anchor_formed_host",          "written_anchor"),
    ("VIRETAN",  "scrambled_control",           "never_written"),
]
UNTOUCHED = ["TAVIREN", "ZEPHYRA", "VIRETAN"]     # rows NOT touched by forge
FORGE_NAME = "QELVARO"

REGISTERED = {
    "question_verbatim":
        "x26 — THE FORGED AUTHOR (R73 ideator card 2; now T292's CAUSAL "
        "PROBE for Law 3's mechanism noun).",
    "background_verbatim":
        "x24 found only gradient-written names lift under displacement; x25 "
        "just proved the lift is NOT the write's mass (it survives and "
        "doubles under the scalpel) — but CAUSALITY is unresolved: does "
        "WRITING cause the fragility, or do writes land on pre-disposable "
        "slots (SELECTION)? THE FORGED AUTHOR is the minimal causal "
        "intervention: ascend a robust never-written name's "
        "initial-character row in the lm_head ONLY — 384 parameters, zero "
        "context write, zero room write — then re-run the displacement "
        "panel. If forging lifts the slot, authorship is LOCAL "
        "ROW-GRADIENT HISTORY (causality in its minimal form, and era-1's "
        "token-row doctrine reconnects through the fragility where x22 "
        "denied it in the knob). If the forge is inert, authorship requires "
        "the full contextual write — and e330 (the writing test) splits "
        "full-causality from selection.",
    "bars_verbatim": {
        "FORGE_LIFTS":
            "the forged QELVARO's complement-lift at host rises >= 3x over "
            "its committed 0.50x band (i.e. >= 1.5x) while the untouched "
            "names stay in their bands — AUTHORSHIP IS LOCAL ROW-GRADIENT "
            "HISTORY; causality lands in its minimal form; era-1's doctrine "
            "reconnects via the fragility; Law 3's noun finalizes as "
            "WRITING-CAUSED (row-local component).",
        "FORGE_INERT":
            "the forged QELVARO stays in the never-written band (<= 2x of "
            "0.50x) despite its raised prior — authorship is constitutively "
            "CONTEXTUAL (the write at contexts, not the row); e330 becomes "
            "the decisive cell; the selection hypothesis strengthens.",
    },
    "prediction_verbatim":
        "Register P-x26a BEFORE compute. Lab guess — the ledger's lesson "
        "applied, REGISTER THE COUNTER-LEAN: after nine straight "
        "mechanism-miss losses the lab declines a confident guess; the "
        "honest prior from x22 (the row delivered 0.8% of the knob) leans "
        "INERT, and x25's doubling (removing mass raised the lift) argues "
        "the slot's disposition is NOT in the head — state your own read if "
        "you differ; predictions are scored.",
    "executor_read": {
        "P_x26a_exec":
            "INERT — the executor does not differ on the word; own "
            "reasoning registered pre-compute: (i) x25's rider showed the "
            "lifted slots couple to GENERIC energy (TAVIREN's "
            "gaussian/complement ratios 0.149 host / 0.680 neutral) — a "
            "192-coordinate OUTPUT-row change cannot re-create a trunk-side "
            "bearer disposition; (ii) the row sits after ln_f, downstream "
            "of the residual bearer x19 measured; (iii) x22's knob reading "
            "(the row delivered 0.8% of the knob). Scores TRUE iff the "
            "verdict == FORGE-INERT.",
    },
    "clauses_fixed": {
        "read_currency":
            "a name's read := p(name[0]) at the final context position, "
            "mean over the site's 60 contexts (x15/x23/x24/x25's committed "
            "currency)",
        "primary_lift":
            "displift(core, disp, name, site) := p(core+disp, name, site) "
            "/ max(p(core, name, site), 1e-12) — the same-organism "
            "displacement-lift (x25's displift rider currency); at "
            "core=base this IS x24's committed lift; the raw x24-currency "
            "lift p(state)/p_base is co-reported for every cell, never "
            "adjudicated (the currency catch: the forge's own ~1.9x prior "
            "bump would otherwise CONFLATE with the displacement effect "
            "and make the INERT bar unreachable by construction)",
        "forge":
            "AdamW(0.9,0.95) wd 0.1, lr 1e-3, house cosine_lr(step-1, "
            "total=400, warmup=100), clip 1.0 (e043's exposure convention, "
            "full-batch deterministic); objective := mean -log p(Q at final "
            "position) over the 60 host-g0 contexts; early-stop at first "
            "step with mean p(Q) >= 0.0050; landing band [0.0040, 0.0120]; "
            "any step above 0.02 -> TEXTURE",
        "row384_catch":
            "the dispatch's '384 parameters' corrected by the organism's "
            "own cfg: n_embd=192 on the 2.74M g1b family — ONE lm_head row "
            "= 192 coordinates; the operative constraint is the design's "
            "own gate (ONLY the target row's coordinates move, bitwise)",
        "untouched_bands":
            "per untouched name, displift(forged, comp, ., host) within "
            "[2/3, 1.5] x its committed x24 host lift; any violation -> "
            "MIXED (attribution broken)",
        "verdicts":
            "on Q_h := displift(forged, comp, QELVARO, host_g0): (1) any "
            "untouched name out of band -> MIXED; (2) FORGE-LIFTS iff Q_h "
            ">= 3 x committed 0.4993 (= 1.4978); (3) FORGE-INERT iff Q_h "
            "<= 2 x committed (= 0.9985); (4) between -> GAP (e330 "
            "decisive)",
        "control_clause":
            "RANDOM-TOUCH-GENERIC iff displift(rnd, comp, QELVARO, host) "
            "also >= the 1.5x bar (row-touch-generic); else ALIGNED-ONLY — "
            "co-reported, never a dispatch bar",
        "gaussian_draw":
            "x25's own seed-25001 draw rebuilt bit-identically (NOT fresh) "
            "— the dispatch's bit-exact x25-repro gate pins the draw",
        "precedence":
            "adjudicate on exactly the frozen clauses; no noun change "
            "without a new registered cell",
    },
    "registration":
        "question + background + design + bars + P-x26a VERBATIM from the "
        "dispatch letter; operationalizations frozen HERE at birth BEFORE "
        "compute; this script committed at birth; adjudicate against "
        "exactly this; no bar shopping.",
}

DEVIATIONS = [
    "CPU-ONLY cell (the dispatch's own feasibility ruling: the row is 192 "
    "trainable coordinates, full-batch 60x130 gradient steps run in "
    "seconds on CPU) — CUDA_VISIBLE_DEVICES='' forced before torch import, "
    "torch threads 4, pocketfft workers 4, no GPU code path, no "
    "envelope-log writes, no other runs/ touched.",
    "THE 384-CATCH (disclosed): the dispatch's '384 parameters' corrected "
    "by the organism's own cfg — the 2.74M g1b family is n_embd=192, so "
    "ONE lm_head row is 192 coordinates; the design's operative constraint "
    "(its own gate) is 'ONLY the target row's coordinates move', verified "
    "bitwise; the wte row is NOT touched ('in the lm_head ONLY').",
    "THE TARGET-BAND CATCH (disclosed): x24's committed QELVARO prior at "
    "host (0.0025906) ALREADY sits inside the dispatch's [0.002, 0.005] "
    "regime, so 'ascend to the regime' is operationalized as ascend to the "
    "TOP of the band — early-stop at the first step with mean p(Q) at host "
    ">= 0.0050 (~a 1.9x prior lift, 'a LIFT to visibility, not a read'); "
    "landing band [0.0040, 0.0120]; any step above 0.02 halts to TEXTURE.",
    "THE CURRENCY CATCH (disclosed, load-bearing): x24's raw lift currency "
    "would CONFLATE the forge's own ~1.9x prior bump with the displacement "
    "effect — the INERT bar would be unreachable by construction; the "
    "primary adjudication currency is the same-organism DISPLACEMENT-LIFT "
    "p(core+disp)/p(core) (x25's displift rider currency); raw x24-currency "
    "lifts co-reported for every cell.",
    "The gaussian rider draw is x25's OWN (seed 25001) rebuilt "
    "bit-identically — NOT fresh — because the dispatch's own gate demands "
    "the base column reproduce x25's committed values bit-exact; the draw "
    "is md5-lineage-bound thereby. This cell's only fresh draw is the "
    "random-row control (seed 26001).",
    "The ascent objective is the panel's own read (mean log p(Q) at the "
    "final position over the 60 host-g0 contexts) — the most favorable "
    "case for FORGE-LIFTS (a strong test of INERT); e043's exposure "
    "convention ported to the row (AdamW (0.9,0.95) wd 0.1, lr 1e-3, house "
    "cosine warmup 100, clip 1.0; full-batch = the deterministic limit, no "
    "sampling RNG); with early stop at ~tens of steps the schedule's "
    "operative segment is the warmup ramp.",
    "The forge's in-room SRCT overlap is an honesty rider at chance "
    "(~0.0604) — 'zero room write' is by construction (no room projection "
    "applied); co-reported, never gated.",
    "QELVARO's other six rows co-reported (host probabilities + "
    "displacement-lifts under the forged state, from the same site "
    "passes); never a separate ascent arm (the design's own clause).",
    "The anchor ZEPHYRA is the K10K complement's parent write's own target "
    "name — the x24-disclosed confound carried: its lift may be the "
    "displacement's own content; TAVIREN is the sole clean written datum "
    "(co-reported bands, never adjudicated on).",
    "CE_R sanity band [1.0, 2.2] scoped to the UNDISPLACED states "
    "(base, forged, rnd); displaced states' ce_r are riders.",
    "gm12 NOT read (the dispatch froze the site axis at host-g0 + neutral, "
    "x24/x25's convention).",
    "n=1 per cell (one lineage, one session, one control draw — the "
    "g-series standing lottery note); no NOTES/THINKING/QUEUE/STATE edits "
    "(dispatch; the heartbeat folds).",
    "Smoke (X26_SMOKE=1): full gate path + the full ascent + all reads "
    "live at every state, own smoke dir; NOTHING adjudicated.",
]

METRICS: dict = {
    "experiment": "x26_forged_author",
    "phase":
        "THE FORGED AUTHOR: ascend the never-written robust name QELVARO's "
        "initial-char lm_head row ONLY (192 coordinates, CPU) to the top of "
        "the marginal-prior band, then re-run x25's displacement panel "
        "(complement + gaussian) on {base, forged, random-row-control} — "
        "FORGE-LIFTS vs FORGE-INERT for Law 3's mechanism noun "
        "(row-gradient causality vs contextual-write constitution)",
    "date": now_utc(),
    "status": "PARTIAL: startup (bars + P-x26a registered at birth)",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": 4},
    "envelope": {
        "device": "CPU ONLY (CUDA_VISIBLE_DEVICES=''; the row-ascent is "
                  "192 parameters — the dispatch's CPU ruling; no "
                  "envelope-log writes)",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": DEVIATIONS,
    "builds_on": [
        "x24 (THE PRIOR-FRAGILITY CENSUS: the 11.8x TAVIREN lift + the "
        "0.50x QELVARO band + the exact panel/site/read harness this cell "
        "ports; the committed numbers this cell reproduces bit-exact)",
        "x25 (THE SLOT'S AUTOPSY: PLASTICITY decisive — the lift survives "
        "and doubles under the scalpel; the displift rider currency + the "
        "gaussian rider draw (seed 25001) + the rider verdict GAP-with-"
        "structure this cell inherits)",
        "T292 (the open causality door: writing-causes vs slot-pre-"
        "disposition — this cell is its registered CAUSAL PROBE)",
        "R73 (ideator card 2: the forged-author design)",
        "e043 (the row-surgery install instrument — the ascent convention "
        "ported: AdamW (0.9,0.95) wd 0.1, lr 1e-3, house cosine, clip 1.0)",
        "T016/e041 (the 384-parameter row-surgery lineage: address-plus-"
        "body — the row is the address, never the body)",
        "x22 (THE KNOB'S SEAT: the row delivered 0.8% of the knob — the "
        "INERT lean's prior)",
        "x19 (THE CORPSE'S DIAL: the bearer-coupling the INERT lean "
        "extends — the row sits downstream of the bearer)",
        "x15 (the K10K dose-matched complement: the construction, the "
        "carrier, the G_REGEN bit-bind convention)",
        "e261/e264 (the K10K write + the room the complement is orthogonal "
        "to)",
        "era-1 doctrine (lm_head token-row directions as causal training "
        "coordinates — the reconnection FORGE-LIFTS would license)",
    ],
    "whats_new": [
        "THE FIRST CAUSAL AUTHORSHIP INTERVENTION: a never-written robust "
        "name's output row given gradient-written history ALONE (192 "
        "coordinates, zero context write, zero room write) — the minimal "
        "causal test of T286/T292's authorship claim",
        "the random-direction matched-norm row control — the ascent's "
        "ALIGNMENT vs any row touch (never before separated)",
        "the full {complement, gaussian} x {base, forged, control} x 4 "
        "names x 2 sites panel with BOTH parents' committed columns "
        "reproduced bit-exact (x24's complement + x25's gaussian) in one "
        "table",
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
@torch.no_grad()
def site_read(net, ids: torch.Tensor, bs: int = 30) -> dict:
    """One forward pass over a site's 60 contexts (x24/x25's exact numerics —
    the committed panel values must reproduce bit-exact)."""
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
    """A name's read from a site pass (x24's battery_read statistics)."""
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
    """x15's form: ||Pv||^2 / ||v||^2."""
    e = float(v64 @ v64)
    if e == 0.0:
        return 0.0
    p = room.project(v64)
    return float((p @ p) / e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-30), 1.0 - 1e-30)
    return float(math.log(p / (1.0 - p)))


@torch.no_grad()
def trunk_final_h(net, ids: torch.Tensor) -> torch.Tensor:
    """The final-position post-ln_f hidden states — TinyGPT.forward's own
    trunk, replicated (the ascent's fixed feature; only the head row
    trains)."""
    B, T = ids.shape
    pos = torch.arange(T)
    x = net.wte(ids) + net.wpe(pos)
    for block in net.h:
        x = block(x)
    return net.ln_f(x)[:, -1, :]


def texture(halt: str) -> SystemExit:
    METRICS["verdict"] = {"word": "TEXTURE",
                          "why": f"{halt} — nothing adjudicated"}
    write_partial(f"HALT: TEXTURE ({halt})")
    return SystemExit(f"{halt} — nothing adjudicated")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X26 — THE FORGED AUTHOR (smoke={SMOKE}) -> {RD}")
    METRICS["birth_commit"] = git_head()
    write_partial("startup (bars + prediction registered, committed at "
                  "birth)")
    set_seed(GLOBAL_SEED)          # global init only; one fresh draw exists

    # ============ P0: corpus, panel gates, the two sites ================
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)

    x24m = json.loads(X24_METRICS.read_text(encoding="utf-8"))
    x25m = json.loads(X25_METRICS.read_text(encoding="utf-8"))
    x24_ncand = int(x24m["gates"]["G_NEUTRAL"]["n_candidates"])

    # G_NAMEFREE + G_PANEL (e311/x24/x25's convention, ported)
    e293_src = E293_SOURCE.read_text(encoding="utf-8")
    m = _re.search(r"NAME_BANK\s*=\s*\(([^)]*)\)", e293_src)
    bank_on_disk = tuple(s.strip().strip('"\'') for s in
                         m.group(1).split(",")) if m else ()
    counts = {nm: train_text.count(nm) for nm, *_ in PANEL}
    initials = [nm[0] for nm, *_ in PANEL]
    fresh_bank = [nm for nm, cls, _ in PANEL
                  if cls == "synthetic_count0_fresh_bank"]
    scrambled = next(nm for nm, cls, _ in PANEL
                     if cls == "scrambled_control")
    synth_all = [nm for nm, cls, _ in PANEL
                 if cls.startswith("synthetic") or cls == "scrambled_control"]
    gpanel = {
        "form": "e311/x24/x25's gate convention ported: synthetic/scrambled "
                "names — 7 letters, every char in the 65-char vocab, "
                "count-0 train split, != G1.NAME; the anchor ZEPHYRA == "
                "G1.NAME by construction; initials all distinct; the bank "
                "parsed from e293's md5-bound source",
        "e293_bank_on_disk": list(bank_on_disk),
        "e293_bank_matches_frozen": bool(bank_on_disk == E293_BANK),
        "fresh_bank_members_in_e293_bank":
            {nm: bool(nm in bank_on_disk) for nm in fresh_bank},
        "counts": counts,
        "chars_in_vocab": {nm: all(c in stoi for c in nm)
                           for nm, *_ in PANEL},
        "seven_letters": {nm: len(nm) == 7 for nm in synth_all},
        "not_g1_name": {nm: nm != G1.NAME for nm in synth_all},
        "zephyra_is_g1_name": bool("ZEPHYRA" == G1.NAME),
        "initials": initials,
        "initials_distinct": bool(len(set(initials)) == len(initials)),
        "scrambled_is_permutation":
            bool(sorted(scrambled) == sorted("TAVIREN")),
        "scrambled_not_a_bank_name": bool(scrambled not in E293_BANK),
    }
    gpanel["pass"] = bool(
        bank_on_disk == E293_BANK
        and all(nm in bank_on_disk for nm in fresh_bank)
        and all(counts[nm] == 0 for nm in synth_all)
        and all(gpanel["chars_in_vocab"].values())
        and all(gpanel["seven_letters"].values())
        and all(gpanel["not_g1_name"].values())
        and gpanel["zephyra_is_g1_name"]
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
    log(f"P0: G_PANEL + G_NAMEFREE PASS — 4 names, initials "
        f"{''.join(initials)} distinct; bank on disk == frozen; ZEPHYRA == "
        f"G1.NAME (the anchor, by construction)")

    # the host-g0 site (e261's splice bank verbatim — x24's construction)
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
        "form": "THE HOST-G0 SITE: e261's splice bank verbatim (x24/x25's "
                "construction — the exact bank on which the committed "
                "panel values were read; also the ascent's objective bank)",
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

    # the neutral site (x24's frozen rule, seed 24001)
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
        "form": "THE NEUTRAL SITE: x24's rule VERBATIM — 60 train-split "
                "windows, 130 chars, ending at an uppercase ASCII letter, "
                "host-free in window + 30-char lookahead; "
                "random.Random(24001) shuffle, first 60; n_candidates "
                "gated == x24's committed count",
        "n_candidates": len(cands),
        "x24_committed_n_candidates": x24_ncand,
        "shape": list(neutral_ids.shape),
        "next_char_upper": f"{upper_ok}/60",
        "host_free": f"{hostfree_ok}/60",
        "pass": bool(list(neutral_ids.shape) == [60, 130]
                     and upper_ok == 60 and hostfree_ok == 60
                     and len(cands) == x24_ncand),
    }
    METRICS["gates"]["G_NEUTRAL"] = gneut
    if not gneut["pass"]:
        raise texture(f"G_NEUTRAL FAILURE: {gneut}")
    sites = {"host_g0": host_ids, "neutral": neutral_ids}
    log(f"P0: G_BATTERY + G_NEUTRAL PASS — host mix 19/41 [60,130]; "
        f"neutral {len(cands)} candidates (x24: {x24_ncand}) -> 60")
    write_partial("P0 panel + both sites gated")

    r_eval_x, r_eval_y = G1.val_windows(val_ids,
                                        "".join(itos[int(i)]
                                                for i in val_ids),
                                        60, G1.R_EVAL_SEED)

    # ============ P0b: the parents hard-bound (Rule 12) =================
    x15m = json.loads(X15_METRICS.read_text(encoding="utf-8"))

    def near(a, b, tol):
        return abs(float(a) - float(b)) <= tol

    lit_checks = {
        "x15_s_full": (x15m["dose_match"]["s_full"], X15_S_FULL, 1e-15),
        "x15_comp_l2": (x15m["complement_regenerated"]["l2_fp64"],
                        X15_COMP_L2_64, 1e-12),
        "x24_verdict": (x24m["verdict"]["word"],
                        "NAME-SPECIFIC-STRUCTURE", None),
        "x24_qelvaro_lift_host": (x24m["panel_reads"]["QELVARO"]
                                  ["lift_host_g0"],
                                  X24_QELVARO_LIFT_HOST, 1e-15),
        "x24_qelvaro_prior_host": (x24m["panel_reads"]["QELVARO"]
                                   ["base_host_g0"]["mean_pz"],
                                   X24_QELVARO_PRIOR_HOST, 1e-15),
        "x24_untouched_lifts": (
            {nm: x24m["panel_reads"][nm]["lift_host_g0"]
             for nm in UNTOUCHED}, X24_UNTOUCHED_LIFTS_HOST, None),
        "x25_fork_verdict": (x25m["verdict"]["fork"]["word"],
                             "PLASTICITY", None),
        "x25_rider_verdict": (x25m["verdict"]["rider"]["word"], "GAP", None),
        "x25_gauss_seed": (x25m["gates"]["G_GAUSS"]["seed"], GAUSS_SEED,
                           None),
        "x25_gauss_norm": (x25m["gates"]["G_GAUSS"]["fp64_l2"],
                           GAUSS_NORM, 1e-12),
        "x24_qelvaro_cid": (x24m["panel_reads"]["QELVARO"]["cid"],
                            stoi["Q"], None),
    }

    def lit_ok(v) -> bool:
        if v[2] is None:
            return v[0] == v[1]
        return near(v[0], v[1], v[2])

    binds = [
        ("e001", CKPT_DIR / BASE_CK, BASE_MD5),
        ("e261_K10K_inst_resume", CKPT_DIR / FACT_CK, FACT_MD5),
        ("e264_rooms", CKPT_DIR / ROOMS264_CK, ROOMS264_MD5),
        ("x15_metrics", X15_METRICS, X15_METRICS_MD5),
        ("x15_comp_xFULL_carrier", X15_COMP_CARRIER, X15_COMP_CARRIER_MD5),
        ("x24_metrics", X24_METRICS, X24_METRICS_MD5),
        ("x25_metrics", X25_METRICS, X25_METRICS_MD5),
        ("e293_bank_source", E293_SOURCE, E293_SOURCE_MD5),
        ("e043_ascent_source", E043_SOURCE, E043_SOURCE_MD5),
    ]
    bind_records = {}
    for nm, p, b in binds:
        got = md5of(p)
        bind_records[nm] = {"path": str(p.relative_to(REPO)), "md5": got,
                            "bound": b, "match": got == b}
        if got != b:
            raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} {got} != {b}")
    gendp = {
        "form": "every parent artifact md5-bound (the displacement's "
                "carrier + both parents' committed metrics + the bank "
                "source + e043's ascent-convention source) + committed "
                "references READ FROM THE ARTIFACTS and cross-checked "
                "against frozen literals, never retyped",
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
    log(f"P0b: G_ENDPOINTS PASS — {len(binds)} parents md5-bound; "
        f"{len(lit_checks)} committed references cross-checked from "
        f"artifacts")
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
        "n_embd": int(base_sd["lm_head.weight"].shape[1]),
        "lm_head_shape": list(base_sd["lm_head.weight"].shape),
        "row_width_192_the_384_catch": bool(
            base_sd["lm_head.weight"].shape[1] == N_EMBD),
        "pass": bool([k for k, _ in named_p] == sd_keys
                     and len(sd_keys) == len(named_p)
                     and N == N_PARAMS
                     and base_sd["lm_head.weight"].shape[1] == N_EMBD),
    }
    if not gflat["pass"] or N != N_PARAMS:
        raise texture(f"FLAT-BASIS GATE FAILURE: {gflat}")
    METRICS["gates"]["G_FLATBASIS"] = gflat

    base_evl = G1.evl_load(base_sd)
    base_site = {s: site_read(base_evl, ids) for s, ids in sites.items()}
    ce_base = float(G1.ce_fixed_cpu(base_evl, r_eval_x, r_eval_y))
    del base_evl
    base_reads = {nm: {s: name_read(base_site[s], stoi[nm[0]])
                       for s in sites} for nm, *_ in PANEL}
    gbase = {
        "form": "the base organism's own priors (every panel name banded "
                "<= 0.05 at host-g0, e311's G_BASE convention) + x24's "
                "committed priors reproduced at 1e-12",
        "priors_host_g0": {nm: base_reads[nm]["host_g0"]["mean_pz"]
                           for nm, *_ in PANEL},
        "max_synthetic_prior_host": max(
            base_reads[nm]["host_g0"]["mean_pz"] for nm in synth_all),
        "committed_x24_priors_match": {
            nm: bool(near(base_reads[nm]["host_g0"]["mean_pz"],
                          x24m["panel_reads"][nm]["base_host_g0"]["mean_pz"],
                          1e-12))
            for nm, *_ in PANEL},
        "bar": BASE_BAR,
        "pass": bool(all(base_reads[nm]["host_g0"]["mean_pz"] <= BASE_BAR
                         for nm in synth_all)
                     and all(near(base_reads[nm]["host_g0"]["mean_pz"],
                                  x24m["panel_reads"][nm]
                                  ["base_host_g0"]["mean_pz"], 1e-12)
                             for nm, *_ in PANEL)),
    }
    if not gbase["pass"]:
        raise texture(f"G_BASEPRIOR FAILURE: {gbase}")
    METRICS["gates"]["G_BASEPRIOR"] = gbase
    log(f"P1: G_FLATBASIS + G_BASEPRIOR PASS (N={N}, n_embd={N_EMBD}; "
        "priors "
        + ", ".join(f"{nm}={base_reads[nm]['host_g0']['mean_pz']:.2e}"
                    for nm, *_ in PANEL)
        + "; all == x24's committed priors at 1e-12)")
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
                "certification (x17/x23/x24/x25's convention)",
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
    log(f"P2: G_ROOM PASS — K10K room D/S BIT-BOUND (idem {max(idem):.1e})")
    write_partial("P2 room bit-bound")

    # ============ P3: THE K10K COMPLEMENT (x15 verbatim) =================
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
        "bar": 1e-12,
    }
    ginroom["pass"] = bool(ginroom["scaled_fp64"] <= 1e-12
                           and ginroom["fp32_injected"] <= 1e-12)
    if not (gdose["pass"] and ginroom["pass"]):
        raise texture(f"DOSE/INROOM FAILURE: {gdose} {ginroom}")
    METRICS["gates"]["G_DOSEMATCH"] = gdose
    METRICS["gates"]["G_INROOM"] = ginroom
    log(f"P3: K10K complement rebuilt x15-verbatim (s_full "
        f"{s_full_k:.10f}; in-room energy {ginroom['scaled_fp64']:.1e})")

    # ============ P4: G_REGEN — the complement vs its CARRIER ==========
    ck_k = torch.load(X15_COMP_CARRIER, map_location="cpu",
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
    log("P4: G_REGEN PASS — THE displacement bit-equal to x15's committed "
        "carrier (every key)")
    write_partial("P4 displacement bit-bound to its carrier")

    # ============ P5: THE FORGE (e043's convention, ONE row) ============
    cid = stoi["Q"]
    if cid != int(x24m["panel_reads"]["QELVARO"]["cid"]):
        raise texture(f"cid mismatch: stoi['Q']={cid} vs x24 "
                      f"{x24m['panel_reads']['QELVARO']['cid']}")

    forge_net = G1.evl_load(base_sd)          # plain load (no anchors)
    for p in forge_net.parameters():          # the trunk is FROZEN; only
        p.requires_grad_(False)               # the row leaf trains below

    h_host = trunk_final_h(forge_net, sites["host_g0"])      # [60, 192]
    h_neut = trunk_final_h(forge_net, sites["neutral"])
    W_all = base_sd["lm_head.weight"]                        # [65, 192]
    logits0_host = h_host @ W_all.T                          # [60, 65] const
    logits0_neut = h_neut @ W_all.T

    def _pQ(h, logits0, row_vec):
        lg = logits0.clone()
        lg[:, cid] = h @ row_vec
        return F.softmax(lg, -1)[:, cid]

    row0 = W_all[cid].clone()
    row = torch.nn.Parameter(row0.clone())
    opt = torch.optim.AdamW([row], lr=FORGE_LR, betas=FORGE_BETAS,
                            weight_decay=FORGE_WD)
    trajectory = [{"step": 0, "lr": 0.0,
                   "p_host": float(_pQ(h_host, logits0_host, row.detach())
                                   .mean()),
                   "p_neutral": float(_pQ(h_neut, logits0_neut,
                                          row.detach()).mean()),
                   "row_delta_l2": 0.0}]
    steps_taken = 0
    halted_overshoot = False
    for step in range(1, FORGE_MAX_STEPS + 1):
        f = cosine_lr(step - 1, FORGE_MAX_STEPS, FORGE_WARMUP)
        for g in opt.param_groups:
            g["lr"] = FORGE_LR * f
        lg = logits0_host.clone()
        lg[:, cid] = h_host @ row
        p_q = F.softmax(lg, -1)[:, cid]
        loss = -(p_q.clamp_min(1e-30).log()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([row], FORGE_CLIP)
        opt.step()
        steps_taken = step
        with torch.no_grad():
            ph = float(_pQ(h_host, logits0_host, row.detach()).mean())
            pn = float(_pQ(h_neut, logits0_neut, row.detach()).mean())
            dl = float(torch.norm(row.detach() - row0))
        trajectory.append({"step": step, "lr": FORGE_LR * f,
                           "p_host": ph, "p_neutral": pn,
                           "row_delta_l2": dl})
        if ph > FORGE_HALT_P:
            halted_overshoot = True
            break
        if ph >= FORGE_STOP_P:
            break
    p_forge_host = trajectory[-1]["p_host"]
    row_final = row.detach().clone()
    ROW_DELTA_L2 = float(torch.norm(row_final - row0))

    if halted_overshoot:
        raise texture(f"THE FORGE OVERSHOT THE REGIME (p_host {p_forge_host}"
                      f" > {FORGE_HALT_P}) — the forge became a read; "
                      f"nothing adjudicated")
    gforge = {
        "form": "THE FORGE: e043's exposure convention ported to ONE row — "
                "AdamW(0.9,0.95) wd 0.1, lr 1e-3, house cosine_lr(step-1, "
                f"total={FORGE_MAX_STEPS}, warmup={FORGE_WARMUP}), clip "
                "1.0; objective := mean -log p(Q) at the final position "
                "over the 60 host-g0 contexts (full-batch, deterministic); "
                f"early-stop at p >= {FORGE_STOP_P} (the top of the "
                "dispatch's [0.002, 0.005] regime — the target-band catch); "
                "the row's change normed + disclosed (the ROW-NORM LADDER "
                "in metrics + figure)",
        "target_row": "lm_head.weight[stoi['Q']] (cid "
                      f"{cid}); width {N_EMBD} (the 384-catch disclosed)",
        "base_p_host": trajectory[0]["p_host"],
        "base_p_neutral": trajectory[0]["p_neutral"],
        "steps_taken": steps_taken,
        "max_steps": FORGE_MAX_STEPS,
        "stop_rule": f"first step with mean p(Q) at host >= {FORGE_STOP_P}",
        "final_p_host": p_forge_host,
        "final_p_neutral": trajectory[-1]["p_neutral"],
        "landing_band": list(FORGE_BAND),
        "prior_lift_from_forge_alone": p_forge_host
                                       / max(trajectory[0]["p_host"],
                                             PRIOR_FLOOR),
        "row_delta_l2_fp32": ROW_DELTA_L2,
        "row_delta_l2_relative": ROW_DELTA_L2
                                 / float(torch.norm(row0)),
        "row_ladder_n_rungs": len(trajectory),
        "pass": bool(FORGE_BAND[0] <= p_forge_host <= FORGE_BAND[1]
                     and 1 <= steps_taken <= FORGE_MAX_STEPS
                     and ROW_DELTA_L2 > 0.0),
    }
    if not gforge["pass"]:
        raise texture(f"G_FORGE FAILURE: {gforge}")
    METRICS["gates"]["G_FORGE"] = gforge
    METRICS["forge"] = {"trajectory": trajectory, **{
        k: v for k, v in gforge.items() if k != "form"}}
    log(f"P5: THE FORGE — {steps_taken} steps (lr ramp "
        f"{trajectory[1]['lr']:.1e}->{trajectory[-1]['lr']:.1e}); p(Q) "
        f"host {trajectory[0]['p_host']:.6f} -> {p_forge_host:.6f} "
        f"(x{p_forge_host / trajectory[0]['p_host']:.2f}); neutral "
        f"{trajectory[0]['p_neutral']:.6f} -> "
        f"{trajectory[-1]['p_neutral']:.6f}; row delta L2 "
        f"{ROW_DELTA_L2:.6f} (row L2 {float(torch.norm(row0)):.4f})")

    # ---- the forged organism + bitwise accounting ----------------------
    forged_sd = {k: v.clone() for k, v in base_sd.items()}
    forged_sd["lm_head.weight"] = W_all.clone()
    forged_sd["lm_head.weight"][cid] = row_final

    def row_accounting(sd32: dict, tag: str) -> dict:
        """Bitwise: ONLY the target row's coordinates move (the design's
        own gate)."""
        others_ok = all(torch.equal(sd32[k], base_sd[k])
                        for k in sd_keys if k != "lm_head.weight")
        dw = (sd32["lm_head.weight"] - base_sd["lm_head.weight"])
        nz = torch.nonzero(dw)
        rows_moved = sorted(set(nz[:, 0].tolist()))
        n_changed = int((dw != 0).sum().item())
        return {
            "form": f"{tag}: exactly the target row's "
                    f"{N_EMBD} coordinates differ; every other key "
                    f"bit-identical",
            "rows_moved": rows_moved,
            "expected_row": [cid],
            "n_elements_changed": n_changed,
            "expected_elements": N_EMBD,
            "other_keys_bit_identical": bool(others_ok),
            "wte_row_untouched": bool(
                torch.equal(sd32["wte.weight"], base_sd["wte.weight"])),
            "pass": bool(rows_moved == [cid] and n_changed == N_EMBD
                         and others_ok),
        }

    gacct_f = row_accounting(forged_sd, "the forged organism")
    if not gacct_f["pass"]:
        raise texture(f"G_FORGEACCOUNT FAILURE: {gacct_f}")
    METRICS["gates"]["G_FORGEACCOUNT"] = gacct_f

    # ---- the random-direction matched-norm control ---------------------
    rrng = np.random.default_rng(ROWRND_SEED)
    rnd64 = rrng.standard_normal(N_EMBD)
    rnd64 = rnd64 * (float(ROW_DELTA_L2) / float(np.linalg.norm(rnd64)))
    rnd_row = torch.from_numpy(
        np.ascontiguousarray(rnd64)).float().reshape(N_EMBD)
    rnd_sd = {k: v.clone() for k, v in base_sd.items()}
    rnd_sd["lm_head.weight"] = W_all.clone()
    rnd_sd["lm_head.weight"][cid] = W_all[cid] + rnd_row
    gacct_r = row_accounting(rnd_sd, "the random-row control")
    RND_DELTA_L2_32 = float(torch.norm(
        rnd_sd["lm_head.weight"][cid] - W_all[cid]))
    gmatch = {
        "form": "the control's fp32 row-delta norm matched to the forge's "
                f"at {ROWMATCH_TOL:g} relative (seed {ROWRND_SEED}, the "
                "cell's ONLY fresh draw)",
        "forged_delta_l2_fp32": ROW_DELTA_L2,
        "rnd_delta_l2_fp32": RND_DELTA_L2_32,
        "rel_diff": abs(RND_DELTA_L2_32 - ROW_DELTA_L2) / ROW_DELTA_L2,
        "bar": ROWMATCH_TOL,
        "pass": bool(abs(RND_DELTA_L2_32 - ROW_DELTA_L2) / ROW_DELTA_L2
                     <= ROWMATCH_TOL and gacct_r["pass"]),
    }
    if not gmatch["pass"]:
        raise texture(f"G_ROWMATCH FAILURE: {gmatch}")
    METRICS["gates"]["G_ROWMATCH"] = gmatch
    METRICS["gates"]["G_FORGEACCOUNT_RND"] = gacct_r

    # ---- the forge's room-overlap honesty rider ------------------------
    def flat_of(sd32: dict) -> np.ndarray:
        return np.concatenate([sd32[k].double().numpy().reshape(-1)
                               for k in sd_keys])

    forge_delta_flat = flat_of(forged_sd) - flat_of(base_sd)
    METRICS["forge"]["inroom_norm_frac_rider"] = inroom_frac(
        room, forge_delta_flat.astype(np.float64))
    METRICS["forge"]["inroom_chance"] = math.sqrt(ROOM_K / N_PARAMS)
    log(f"P5: G_FORGE + G_FORGEACCOUNT + G_ROWMATCH PASS — only row {cid}'s"
        f" {N_EMBD} coords moved (bitwise); control norm "
        f"{RND_DELTA_L2_32:.6f} vs forge {ROW_DELTA_L2:.6f}; in-room rider "
        f"{METRICS['forge']['inroom_norm_frac_rider']:.6f} (chance "
        f"{METRICS['forge']['inroom_chance']:.6f})")
    write_partial("P5 the forge + control constructed")

    # ============ P6: THE GAUSSIAN (x25's own draw, rebuilt) ============
    grng = np.random.default_rng(GAUSS_SEED)
    gauss64 = grng.standard_normal(N_PARAMS)
    gauss64 = gauss64 * (GAUSS_NORM / float(np.linalg.norm(gauss64)))
    gauss_inroom_norm = inroom_frac(room, gauss64)
    gauss_inroom_energy = inroom_energy_share(room, gauss64)
    gauss32 = {k: v.float()
               for k, v in unflat_like_base(gauss64).items()}
    fl_g32 = np.concatenate([gauss32[k].double().numpy().reshape(-1)
                             for k in sd_keys])
    cos_gc = float(gauss64 @ scaled_k64
                   / (np.linalg.norm(gauss64) * np.linalg.norm(scaled_k64)))
    ggauss = {
        "form": "x25's OWN gaussian draw (seed 25001) rebuilt "
                "bit-identically (NOT fresh — the dispatch's bit-exact "
                "x25-repro gate pins the draw), fp64-normalized to the "
                "matched ~9.18 dose, fp32-injected by x15's method; "
                "in-room fractions gated at chance",
        "seed": GAUSS_SEED,
        "fp64_l2": float(np.linalg.norm(gauss64)),
        "fp64_l2_abs_diff": abs(float(np.linalg.norm(gauss64)) - GAUSS_NORM),
        "fp32_l2": float(np.linalg.norm(fl_g32)),
        "inroom_norm_frac": gauss_inroom_norm,
        "inroom_norm_band": list(GAUSS_INROOM_NORM_BAND),
        "chance_norm_frac": math.sqrt(ROOM_K / N_PARAMS),
        "inroom_energy_share": gauss_inroom_energy,
        "inroom_energy_band": list(GAUSS_INROOM_ENERGY_BAND),
        "chance_energy_share": ROOM_K / N_PARAMS,
        "cos_vs_complement": cos_gc,
        "pass": bool(abs(float(np.linalg.norm(gauss64)) - GAUSS_NORM) <= 1e-12
                     and GAUSS_INROOM_NORM_BAND[0] <= gauss_inroom_norm
                     <= GAUSS_INROOM_NORM_BAND[1]
                     and GAUSS_INROOM_ENERGY_BAND[0] <= gauss_inroom_energy
                     <= GAUSS_INROOM_ENERGY_BAND[1]),
    }
    if not ggauss["pass"]:
        raise texture(f"G_GAUSS FAILURE: {ggauss}")
    METRICS["gates"]["G_GAUSS"] = ggauss
    log(f"P6: G_GAUSS PASS — norm {ggauss['fp64_l2']:.10f}; in-room norm "
        f"frac {gauss_inroom_norm:.6f} (chance "
        f"{ggauss['chance_norm_frac']:.6f}); cos vs comp {cos_gc:+.2e}")
    write_partial("P6 gaussian rebuilt (x25's draw)")

    # ============ P7: THE STATES — one applied net, many probes =========
    comp_delta = {k: ck_k["delta"][k].float() for k in sd_keys}

    def displaced(sd32: dict, delta: dict) -> dict:
        return {k: sd32[k] + delta[k] for k in sd_keys}

    states: dict = {}
    states["base"] = {"sd": base_sd,
                      "source": "in-memory e001 (md5-bound)"}
    states["live_comp"] = {
        "sd": None,   # THE carrier's own model dict (x24's G_ONESTATE path)
        "source": "runs/x15/x15_comp_xFULL.pt ['model'] (md5-bound; "
                  "bit-regenerated + verified) — x24's own comp state"}
    states["live_gauss"] = {
        "sd": displaced(base_sd, gauss32),
        "source": "base + gaussian (x25's own draw, fp32, x15's injection)"}
    states["forged"] = {
        "sd": forged_sd,
        "source": f"base + row-ascent on lm_head row {cid} (the forge; "
                  f"{steps_taken} steps, row delta L2 {ROW_DELTA_L2:.6f})"}
    states["rnd"] = {
        "sd": rnd_sd,
        "source": f"base + random-direction row delta at matched norm "
                  f"({RND_DELTA_L2_32:.6f}; seed {ROWRND_SEED})"}
    for core, tag in (("forged", "THE FORGED ARM"),
                      ("rnd", "THE CONTROL ARM")):
        states[f"{core}_comp"] = {
            "sd": displaced(states[core]["sd"], comp_delta),
            "source": f"{core} + complement ({tag})"}
        states[f"{core}_gauss"] = {
            "sd": displaced(states[core]["sd"], gauss32),
            "source": f"{core} + gaussian ({tag} rider)"}

    for key, st in states.items():
        if key == "base":                      # reuse P1's probe of e001
            st["site_passes"] = base_site
            st["ce_r"] = ce_base
            log(f"P7: state {key:12s} read (ce_r {st['ce_r']:.3f}; P1 "
                f"reuse)")
            continue
        sd = st["sd"] if st["sd"] is not None else ck_k["model"]
        net = G1.evl_load(sd)
        st["site_passes"] = {s: site_read(net, ids)
                             for s, ids in sites.items()}
        st["ce_r"] = float(G1.ce_fixed_cpu(net, r_eval_x, r_eval_y))
        del net
        log(f"P7: state {key:12s} read (ce_r {st['ce_r']:.3f})")

    gone = {
        "form": "every probed state is ONE applied net (constructed once, "
                "probed at both sites on all 4 names; the live complement "
                "panel probed from the carrier's own model dict) — no "
                "per-name reconstruction; identical by construction, "
                "provenance recorded",
        "n_states": len(states),
        "pass": True,
    }
    METRICS["gates"]["G_ONESTATE"] = gone

    # ============ P8: the load/repro gates ==============================
    # G_FORGEREAD: the ascent's final measured p vs the site-read p
    p_site = float(states["forged"]["site_passes"]["host_g0"]
                   ["probs"][:, cid].mean())
    gforgeread = {
        "form": "the ascent's final measured mean p(Q) at host vs the "
                "standard site-read path on the same state (different code "
                "paths, same math) at 1e-6",
        "ascent_final": p_forge_host, "site_read": p_site,
        "abs_diff": abs(p_forge_host - p_site), "bar": FORGEREAD_TOL,
        "pass": bool(abs(p_forge_host - p_site) <= FORGEREAD_TOL),
    }
    if not gforgeread["pass"]:
        raise texture(f"G_FORGEREAD FAILURE: {gforgeread}")
    METRICS["gates"]["G_FORGEREAD"] = gforgeread

    # G_X24REPRO: the live comp panel reproduces x24 exactly
    repro24 = {}
    for nm, *_ in PANEL:
        for s in sites:
            col = "comp_host_g0" if s == "host_g0" else "comp_neutral"
            mine = float(states["live_comp"]["site_passes"][s]
                         ["probs"][:, stoi[nm[0]]].mean())
            committed = x24m["panel_reads"][nm][col]["mean_pz"]
            repro24[f"{nm}_{s}"] = {"mine": mine, "committed": committed,
                                    "abs_diff": abs(mine - committed)}
    gx24 = {
        "form": "the live panel's (base + the K10K complement, the x15 "
                "carrier's own model) values reproduce x24's committed "
                "numbers EXACTLY (|d| <= 1e-12) — read from the md5-bound "
                "x24 metrics, never retyped",
        "cells": repro24,
        "bar": REPRO_TOL,
        "pass": bool(all(v["abs_diff"] <= REPRO_TOL
                         for v in repro24.values())),
    }
    if not gx24["pass"]:
        raise texture(f"G_X24REPRO FAILURE: {gx24}")
    METRICS["gates"]["G_X24REPRO"] = gx24

    # G_X25REPRO: the live gaussian panel reproduces x25 exactly
    repro25 = {}
    for nm, *_ in PANEL:
        for s in sites:
            col = "live_gauss_host_g0" if s == "host_g0" \
                else "live_gauss_neutral"
            mine = float(states["live_gauss"]["site_passes"][s]
                         ["probs"][:, stoi[nm[0]]].mean())
            committed = x25m["panel_reads"][nm][col]["mean_pz"]
            repro25[f"{nm}_{s}"] = {"mine": mine, "committed": committed,
                                    "abs_diff": abs(mine - committed)}
    gx25 = {
        "form": "the live gaussian panel (base + x25's own seed-25001 draw, "
                "rebuilt) reproduces x25's committed live_gauss values "
                "EXACTLY (|d| <= 1e-12) — read from the md5-bound x25 "
                "metrics, never retyped",
        "cells": repro25,
        "bar": REPRO_TOL,
        "pass": bool(all(v["abs_diff"] <= REPRO_TOL
                         for v in repro25.values())),
    }
    if not gx25["pass"]:
        raise texture(f"G_X25REPRO FAILURE: {gx25}")
    METRICS["gates"]["G_X25REPRO"] = gx25

    # G_HOSTSANITY: e318's band on the UNDISPLACED states
    san_states = ["base", "forged", "rnd"]
    ces = {k: states[k]["ce_r"] for k in san_states}
    gsan = {
        "form": "ce_r within [1.0, 2.2] at the three UNDISPLACED states "
                "(base, forged, rnd — e318's band: a forged organism that "
                "torches the corpus read flags the kill-everything "
                "confound); displaced states' ce_r are riders",
        "ce_r": ces, "band": list(CE_BAND),
        "pass": bool(all(CE_BAND[0] <= v <= CE_BAND[1] for v in ces.values())),
    }
    if not gsan["pass"]:
        raise texture(f"G_HOSTSANITY FAILURE: {gsan}")
    METRICS["gates"]["G_HOSTSANITY"] = gsan
    log(f"P8: G_FORGEREAD (|d| {gforgeread['abs_diff']:.1e}) + G_X24REPRO "
        f"(max |d| {max(v['abs_diff'] for v in repro24.values()):.1e}) + "
        f"G_X25REPRO (max |d| "
        f"{max(v['abs_diff'] for v in repro25.values()):.1e}) + "
        f"G_HOSTSANITY PASS — both parents' columns bit-faithful")
    write_partial("P8 load/repro gates PASSED (x24 + x25 panels reproduced)")

    # ============ P9: the census table ===================================
    reads = {}
    for nm, cls, role in PANEL:
        cidn = stoi[nm[0]]
        reads[nm] = {"class": cls, "role": role, "cid": cidn}
        for st in states:
            for s in sites:
                reads[nm][f"{st}_{s}"] = name_read(
                    states[st]["site_passes"][s], cidn)
    for nm in reads:
        for s in sites:
            pb = reads[nm][f"base_{s}"]["mean_pz"]
            for st in states:
                if st == "base":
                    continue
                ps = reads[nm][f"{st}_{s}"]["mean_pz"]
                reads[nm][f"lift_{st}_{s}"] = ps / max(pb, PRIOR_FLOOR)
                reads[nm][f"dlogit_{st}_{s}"] = _logit(ps) - _logit(pb)
            # the PRIMARY currency: same-organism displacement-lifts
            for disp in ("comp", "gauss"):
                for core in ("base", "forged", "rnd"):
                    und = reads[nm][f"{core}_{s}"]["mean_pz"]
                    dis = reads[nm][
                        f"live_{disp}_{s}" if core == "base"
                        else f"{core}_{disp}_{s}"]["mean_pz"]
                    reads[nm][f"displift_{core}_{disp}_{s}"] = \
                        dis / max(und, PRIOR_FLOOR)
    METRICS["panel_reads"] = reads

    # the other-rows co-report (the design's clause; never an arm)
    other_rows = {}
    for ch in FORGE_NAME:
        c = stoi[ch]
        if ch == "Q":
            continue
        row = {"cid": c}
        for st in ("base", "forged", "forged_comp"):
            row[f"p_host_{st}"] = float(
                states[st]["site_passes"]["host_g0"]["probs"][:, c].mean())
        row["displift_forged_comp_host"] = (
            row["p_host_forged_comp"] / max(row["p_host_forged"],
                                            PRIOR_FLOOR))
        other_rows[ch] = row
    METRICS["qelvaro_other_rows_correport"] = other_rows
    write_partial("P9 the census read (9 states x 2 sites x 4 names)")

    if SMOKE:
        METRICS["status"] = ("SMOKED — full gate path + the full ascent + "
                             "all reads exercised at every state; NOTHING "
                             "adjudicated")
        write_partial("SMOKE COMPLETE — nothing adjudicated")
        log("SMOKE COMPLETE (nothing adjudicated)")
        raise SystemExit(0)

    # ============ P10: ADJUDICATION (frozen) ============================
    committed_q = float(x24m["panel_reads"]["QELVARO"]["lift_host_g0"])
    bar_lifts = FORGE_LIFTS_MULT * committed_q
    bar_inert = FORGE_INERT_MULT * committed_q
    Q_h = reads["QELVARO"]["displift_forged_comp_host_g0"]
    Q_h_neut = reads["QELVARO"]["displift_forged_comp_neutral"]
    untouched_detail = {}
    untouched_ok = True
    for nm in UNTOUCHED:
        v = reads[nm]["displift_forged_comp_host_g0"]
        c = float(x24m["panel_reads"][nm]["lift_host_g0"])
        lo, hi = UNTOUCHED_BAND[0] * c, UNTOUCHED_BAND[1] * c
        untouched_detail[nm] = {"forged_displift": v, "committed": c,
                                "band": [lo, hi],
                                "in_band": bool(lo <= v <= hi)}
        if not (lo <= v <= hi):
            untouched_ok = False

    lifts_numeric = Q_h >= bar_lifts
    inert_numeric = Q_h <= bar_inert
    if not untouched_ok:
        verdict_word = "MIXED"
    elif lifts_numeric:
        verdict_word = "FORGE-LIFTS"
    elif inert_numeric:
        verdict_word = "FORGE-INERT"
    else:
        verdict_word = "GAP"

    # the control clause (co-report, never a dispatch bar)
    Q_r = reads["QELVARO"]["displift_rnd_comp_host_g0"]
    control_word = ("RANDOM-TOUCH-GENERIC" if Q_r >= bar_lifts
                    else "ALIGNED-ONLY")

    verdict_clause = {
        "primary_currency": REGISTERED["clauses_fixed"]["primary_lift"],
        "Q_h_displift_forged_comp_host": Q_h,
        "Q_h_neutral_correport": Q_h_neut,
        "committed_qelvaro_lift": committed_q,
        "forge_lifts_bar": bar_lifts, "forge_inert_bar": bar_inert,
        "lifts_numeric_fired": bool(lifts_numeric),
        "inert_numeric_fired": bool(inert_numeric),
        "untouched_ok": bool(untouched_ok),
        "untouched_detail": untouched_detail,
        "raw_x24_currency_correport": {
            "lift_forged_comp_host":
                reads["QELVARO"]["lift_forged_comp_host_g0"],
            "note": "p(forged+comp)/p_base — conflates the forge's own "
                    f"x{gforge['prior_lift_from_forge_alone']:.2f} prior "
                    "bump; never adjudicated (the currency catch)"},
        "control_clause": {
            "word": control_word,
            "Q_r_displift_rnd_comp_host": Q_r,
            "bar": bar_lifts,
            "note": "co-reported: is it the ascent's alignment or any row "
                    "touch?"},
        "gaussian_rider_correport": {
            "displift_forged_gauss_host":
                reads["QELVARO"]["displift_forged_gauss_host_g0"],
            "displift_base_gauss_host":
                reads["QELVARO"]["displift_base_gauss_host_g0"],
            "displift_rnd_gauss_host":
                reads["QELVARO"]["displift_rnd_gauss_host_g0"]},
        "row_norm_ladder": {
            "steps": steps_taken, "row_delta_l2_fp32": ROW_DELTA_L2,
            "rnd_matched_l2_fp32": RND_DELTA_L2_32,
            "relative_to_row_l2": ROW_DELTA_L2
                                   / float(torch.norm(row0))},
    }
    p_a = {
        "statement": REGISTERED["prediction_verbatim"],
        "P_x26a_lab_guess": "INERT (the counter-lean; the lab declines a "
                            "confident guess)",
        "P_x26a_outcome": ("HIT — the verdict is FORGE-INERT"
                           if verdict_word == "FORGE-INERT" else
                           f"MISS — the verdict is {verdict_word}"),
        "P_x26a_exec_statement":
            REGISTERED["executor_read"]["P_x26a_exec"],
        "P_x26a_exec_outcome": ("HIT — the executor read was INERT"
                                if verdict_word == "FORGE-INERT" else
                                f"MISS — the verdict is {verdict_word}"),
    }
    METRICS["verdict"] = {
        "word": verdict_word,
        "clause": verdict_clause,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "prediction": p_a,
    }
    write_partial(f"P10 ADJUDICATED: {verdict_word} (Q_h {Q_h:.4f}x; "
                  f"bars {bar_inert:.4f}/{bar_lifts:.4f}; control "
                  f"{control_word})")

    # ============ P11: the figure ========================================
    import matplotlib                                 # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                   # noqa: E402

    names = [nm for nm, *_ in PANEL]
    cores = [("base", "base (committed)"), ("forged", "FORGED"),
             ("rnd", "random-row control")]
    disps = [("comp", "complement"), ("gauss", "gaussian")]
    fig = plt.figure(figsize=(16.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.1])
    for pi, s in enumerate(("host_g0", "neutral")):
        ax = fig.add_subplot(gs[0, pi])
        width = 0.26
        xpos = np.arange(len(names))
        for ci, (core, lbl) in enumerate(cores):
            vals = [reads[nm][f"displift_{core}_comp_{s}"] for nm in names]
            ax.bar(xpos + (ci - 1) * width, vals, width, label=lbl,
                   alpha=0.9 if core == "forged" else 0.55,
                   color=("C0" if core == "base"
                          else "C3" if core == "forged" else "C7"))
        ax.set_yscale("log")
        if s == "host_g0":
            ax.axhline(bar_lifts, color="C3", ls="--", lw=1.2)
            ax.axhline(bar_inert, color="k", ls=":", lw=1.2)
            ax.text(len(names) - 0.45, bar_lifts * 1.05,
                    f"FORGE-LIFTS bar {bar_lifts:.2f}x", fontsize=7,
                    color="C3", ha="right")
            ax.text(len(names) - 0.45, bar_inert * 0.93,
                    f"FORGE-INERT ceiling {bar_inert:.2f}x", fontsize=7,
                    color="k", ha="right")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{nm}\n({next(r for n, _, r in PANEL
                                           if n == nm)})"
                            for nm in names], fontsize=7.5)
        ax.set_ylabel("complement displacement-lift  p(core+comp)/p(core)",
                      fontsize=8)
        ax.set_title(f"({'a' if pi == 0 else 'b'}) "
                     f"{'HOST-G0' if s == 'host_g0' else 'NEUTRAL'} — "
                     "the forged vs base vs control panel", fontsize=9.5)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.2, axis="y")
    # (c) the row-norm ladder
    axc = fig.add_subplot(gs[0, 2])
    steps_l = [t["step"] for t in trajectory]
    p_l = [t["p_host"] for t in trajectory]
    dl_l = [t["row_delta_l2"] for t in trajectory]
    axc.plot(steps_l, p_l, "o-", ms=3, color="C3", label="p(Q) at host")
    axc.axhline(FORGE_STOP_P, color="k", ls="--", lw=1,
                label=f"stop target {FORGE_STOP_P}")
    axc.axhspan(FORGE_BAND[0], FORGE_BAND[1], color="C2", alpha=0.12,
                label="landing band")
    axc.set_yscale("log")
    axc.set_xlabel("ascent step", fontsize=8)
    axc.set_ylabel("mean p(Q) at host-g0", fontsize=8, color="C3")
    axc2 = axc.twinx()
    axc2.plot(steps_l, dl_l, "s--", ms=2.5, color="C1",
              label="row delta L2")
    axc2.axhline(RND_DELTA_L2_32, color="C7", ls=":", lw=1.2,
                 label="control rung (matched)")
    axc2.set_ylabel("||row - row_0||_2 (fp32)", fontsize=8, color="C1")
    axc.set_title("(c) THE ROW-NORM LADDER — the forge's ascent",
                  fontsize=9.5)
    h1, l1 = axc.get_legend_handles_labels()
    h2, l2 = axc2.get_legend_handles_labels()
    axc.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower right")
    axc.grid(alpha=0.2)
    fig.suptitle("x26 — THE FORGED AUTHOR: "
                 f"{FORGE_NAME}'s 'Q' lm_head row given gradient history "
                 f"alone ({steps_taken} steps, ||dRow|| {ROW_DELTA_L2:.4f}) "
                 f"— verdict {verdict_word}; control {control_word}",
                 fontsize=11)
    fig.savefig(RD / "x26_forged_author.png", dpi=140, bbox_inches="tight")
    log("FIGURE written")

    # ============ P12: REPORT.md =========================================
    def fmt(v):
        return f"{v:.4g}" if v >= 1e-4 else f"{v:.3e}"

    tbl = ["| state | " + " | ".join(
               f"{nm}: p / displift host | {nm}: displift neutral"
               for nm in names) + " |",
           "|---|" + "---|---|" * len(names)]
    row_states = [("base", None), ("live_comp", ("base", "comp")),
                  ("live_gauss", ("base", "gauss")),
                  ("forged", None), ("forged_comp", ("forged", "comp")),
                  ("forged_gauss", ("forged", "gauss")),
                  ("rnd", None), ("rnd_comp", ("rnd", "comp")),
                  ("rnd_gauss", ("rnd", "gauss"))]
    for st, cd in row_states:
        row = [f"| **{st}**"]
        for nm in names:
            row.append(
                f"{fmt(reads[nm][f'{st}_host_g0']['mean_pz'])} / "
                + ("(core)" if cd is None
                   else f"**{reads[nm][f'displift_{cd[0]}_{cd[1]}_host_g0']:.2f}x**"))
            row.append("(core)" if cd is None
                       else f"{reads[nm][f'displift_{cd[0]}_{cd[1]}_neutral']:.2f}x")
        tbl.append(" | ".join(row) + " |")
    core_tbl = ["| core | disp | QELVARO host | QELVARO neutral | "
                "TAVIREN host | ZEPHYRA host | VIRETAN host |",
                "|---|---|---|---|---|---|---|"]
    for core, lbl in cores:
        for disp, dlbl in disps:
            core_tbl.append(
                f"| {lbl} | {dlbl} | "
                f"**{reads['QELVARO'][f'displift_{core}_{disp}_host_g0']:.3f}x**"
                f" | {reads['QELVARO'][f'displift_{core}_{disp}_neutral']:.3f}x"
                f" | {reads['TAVIREN'][f'displift_{core}_{disp}_host_g0']:.3f}x"
                f" | {reads['ZEPHYRA'][f'displift_{core}_{disp}_host_g0']:.3f}x"
                f" | {reads['VIRETAN'][f'displift_{core}_{disp}_host_g0']:.3f}x"
                " |")
    forge_tbl = ["| rung | value |", "|---|---|",
                 f"| base p(Q) at host (x24's committed prior) | "
                 f"{trajectory[0]['p_host']:.6f} |",
                 f"| ascent steps taken (early stop at p >= "
                 f"{FORGE_STOP_P}) | {steps_taken} |",
                 f"| forged p(Q) at host | {p_forge_host:.6f} "
                 f"(x{p_forge_host / trajectory[0]['p_host']:.2f} prior "
                 f"lift) |",
                 f"| forged p(Q) at neutral (co-measured, not objective) | "
                 f"{trajectory[-1]['p_neutral']:.6f} |",
                 f"| row delta L2 (fp32) | {ROW_DELTA_L2:.6f} "
                 f"(row L2 {float(torch.norm(row0)):.4f}; relative "
                 f"{ROW_DELTA_L2 / float(torch.norm(row0)):.4f}) |",
                 f"| control rung: random-direction row delta L2 | "
                 f"{RND_DELTA_L2_32:.6f} (matched at "
                 f"{gmatch['rel_diff']:.1e}) |",
                 f"| forge in-room SRCT fraction (rider; chance "
                 f"{math.sqrt(ROOM_K / N_PARAMS):.6f}) | "
                 f"{METRICS['forge']['inroom_norm_frac_rider']:.6f} |",
                 f"| QELVARO displift(forged, comp, host) = Q_h | "
                 f"**{Q_h:.4f}x** (bars: <= {bar_inert:.4f} INERT / >= "
                 f"{bar_lifts:.4f} LIFTS) |",
                 f"| QELVARO displift(rnd, comp, host) — the control | "
                 f"{Q_r:.4f}x ({control_word}) |",
                 f"| raw x24-currency lift(forged+comp) — co-report only | "
                 f"{reads['QELVARO']['lift_forged_comp_host_g0']:.4f}x |"]
    other_tbl = ["| QELVARO char | p_host base | p_host forged | "
                 "p_host forged+comp | displift(forged, comp) |",
                 "|---|---|---|---|---|"]
    for ch, rw in other_rows.items():
        other_tbl.append(
            f"| {ch} (cid {rw['cid']}) | {rw['p_host_base']:.6f} | "
            f"{rw['p_host_forged']:.6f} | {rw['p_host_forged_comp']:.6f} | "
            f"{rw['displift_forged_comp_host']:.3f}x |")
    gpass = sum(1 for g in METRICS["gates"].values()
                if isinstance(g, dict) and g.get("pass"))
    gtot = len(METRICS["gates"])
    ce_line = ", ".join(f"{k} {v['ce_r']:.3f}" for k, v in states.items())
    report = f"""# x26 — THE FORGED AUTHOR (verdict {verdict_word}; control {control_word})

**The causal probe (T292):** x24 found only gradient-written names lift
under the full-dose out-of-room displacement; x25 proved the lift is NOT
the write's mass (it survives and doubles under the scalpel). The open
fork: does WRITING cause the fragility, or do writes land on
pre-disposable slots (SELECTION)? **This cell:** the minimal causal
intervention — QELVARO (x24's most robust never-written name, the
23.7x-robust datum) given gradient-written history ALONE: its initial-char
'Q' lm_head row ascended to the top of the marginal-prior band ({steps_taken}
steps, e043's convention ported to {N_EMBD} coordinates — the 384-catch
disclosed), then x25's displacement panel re-run on {{base, forged,
random-row control}}.

**The primary currency (the currency catch, frozen at birth):** the
displacement-lift p(core+disp)/p(core) — same-organism, x25's displift
currency; x24's raw lift would conflate the forge's own
x{p_forge_host / trajectory[0]['p_host']:.2f} prior bump. Both parents'
base columns reproduced bit-exact (max |d| x24
{max(v['abs_diff'] for v in repro24.values()):.1e}, x25
{max(v['abs_diff'] for v in repro25.values()):.1e} <= 1e-12).

## THE FORGE (the row-norm ladder)

{chr(10).join(forge_tbl)}

## THE DISPLACEMENT-LIFT PANEL (the primary currency)

{chr(10).join(core_tbl)}

## THE FULL STATE TABLE (p host / displift host, displift neutral)

{chr(10).join(tbl)}

## QELVARO's other rows (co-report, never an arm)

{chr(10).join(other_tbl)}

## THE VERDICT

- Q_h (displift(forged, comp, QELVARO, host)) = **{Q_h:.4f}x** vs the
  committed {committed_q:.4f}x band.
- FORGE-LIFTS bar (>= 3x committed = {bar_lifts:.4f}x): **{'FIRES' if lifts_numeric else 'does not fire'}**.
- FORGE-INERT bar (<= 2x committed = {bar_inert:.4f}x): **{'FIRES' if inert_numeric else 'does not fire'}**.
- Untouched names in bands ({[f'{nm}: {untouched_detail[nm]["forged_displift"]:.2f}x vs {untouched_detail[nm]["band"][0]:.2f}-{untouched_detail[nm]["band"][1]:.2f}' for nm in UNTOUCHED]}): **{untouched_ok}**.
- Control clause: {control_word} (Q_r {Q_r:.4f}x).

**Verdict: {verdict_word}.** Bars frozen at birth (see metrics);
P-x26a: {p_a['P_x26a_outcome']}; executor read: {p_a['P_x26a_exec_outcome']}.

## Gates: {gpass}/{gtot} PASS

G_ENDPOINTS ({len(binds)} md5 binds incl. both parents' metrics + e043's
ascent source; {len(lit_checks)} committed references read from
artifacts), G_FLATBASIS (+the 384-catch: row width 192), G_PANEL
(+G_NAMEFREE), G_BATTERY, G_NEUTRAL (x24's n_candidates), G_ROOM (D/S
bit-bound), G_ORTH, G_DOSEMATCH (1e-12), G_INROOM (1e-12), G_REGEN (the
displacement bit-equal to x15's carrier), G_GAUSS (x25's draw rebuilt),
G_FORGE (the ascent's target regime + trajectory), G_FORGEACCOUNT (ONLY
row {cid}'s {N_EMBD} coords move, bitwise — forged AND control),
G_ROWMATCH (control norm at {gmatch['rel_diff']:.1e}), G_FORGEREAD (|d|
{gforgeread['abs_diff']:.1e}), G_BASEPRIOR, G_X24REPRO (max |d|
{max(v['abs_diff'] for v in repro24.values()):.1e}), G_X25REPRO (max |d|
{max(v['abs_diff'] for v in repro25.values()):.1e}), G_HOSTSANITY
(ce_r {gsan['ce_r']} in [1.0, 2.2]), G_ONESTATE.

ce_r per state (riders beyond base/forged/rnd): {ce_line}. The displaced
states' organism cost is part of what the panel measures (disclosed).

Disclosures: CPU-only (the dispatch's ruling); the 384-catch (n_embd=192
— the design's own bitwise gate governs); the target-band catch (the
prior already sat in [0.002, 0.005] — the forge ascends to the TOP of the
band, a lift not a read); the currency catch (displift primary, raw
co-reported); the gaussian is x25's own draw rebuilt (the bit-exact gate
pins it); the ascent objective is the panel's own read (the most
favorable case for FORGE-LIFTS); the anchor ZEPHYRA's confound carried
(TAVIREN the sole clean written datum); n=1 per cell (one lineage, one
session, one control draw); bars + P-x26a frozen at birth before any
compute.
"""
    (RD / "REPORT.md").write_text(report, encoding="utf-8")

    METRICS["outputs"] = {
        "figure": f"runs/{NAME}/x26_forged_author.png",
        "metrics": f"runs/{NAME}/metrics.json",
        "report": f"runs/{NAME}/REPORT.md",
    }
    METRICS["status"] = "COMPLETE — adjudicated"
    write_partial("P12 COMPLETE (report + figure)")
    log(f"DONE — verdict {verdict_word} (Q_h {Q_h:.4f}x vs bars "
        f"{bar_inert:.4f}/{bar_lifts:.4f}; committed {committed_q:.4f}x; "
        f"control Q_r {Q_r:.4f}x -> {control_word}; forge {steps_taken} "
        f"steps, row dL2 {ROW_DELTA_L2:.6f})")


if __name__ == "__main__":
    main()
