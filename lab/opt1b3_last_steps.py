"""OPT1B3 — THE LAST FOURTEEN STEPS (T145's deciding door; the bleed's own kill-D).

WHY: the diffusive bleed (SGD lr 1e-2 on the licensed e185 wash cell — the
raw-gradient direction at its own small step size) crossed INTO the kill
window's lower margin ALIVE (g-12 0.324 at D 2.12, ~s1167) and STALLED:
alive 0.3290 at D 2.1455 at opt1b2's frozen 1200-step cap. The last-slope
extrapolation reads the kill ~s1214 — FOURTEEN steps past the cap. T145's
deciding door: a tens-of-steps continuation decides between the bleed's OWN
kill-D (reunifying the gate: protection = staying diffusive/slow) and a
separate ring per trajectory class.

RECOVERY NOTE (disclosed before compute): this is the THIRD dispatch of
this cell, the second recovery. Dispatch-1 died before any artifacts
existed. Dispatch-2 executed the frozen registration: all pre-dispatch
gates PASSED (the s1200 resume certified bit), then the FIFTH disruption
(user-confirmed system shutdown, 2026-10-01) killed its executor
MID-FLIGHT at s1264, after chunk 0's 64 every-step-read steps (s1201..s1264,
all alive, g-12 0.316-0.342, D 2.1918). Its survivors — runs/opt1b3/
chunk_state.pt (mid-flight, stop=None) and the PARTIAL metrics.json
(committed at b7f8e20) — are this dispatch's resume point. THIS dispatch
resumes from that chunk_state @ s1264 IFF the MID-RUN RESUME GATE
(gates.G_MIDRESUME, added by this dispatch: model flat-md5 recompute ==
payload == the partial's chunk-0 tail; journal contiguous s1201..s1264
with one read per step; tails bit-equal the committed partial; the
displacement recompute; stream uniqueness across opt1b+opt1b2+the loaded
rows) certifies it; on failure it ABORTS (control failure). It NEVER
re-executes steps 1201..1264. The registration is the dispatch's letter
VERBATIM (bars quoted word-for-word); no bar shopping; no bar, cadence,
gate threshold or adjudication logic changed by the recovery.

THE CELL (CPU-only, minutes): resume the committed s1200 state via opt1b2's
CERTIFIED path (the chunk_state == s1200.pt == journal-tail gates, all
bit/tolerance bound) and continue up to 150 steps (to s1350) or the kill,
whichever first; g-12 read EVERY step, D(t) per step, CE_R every 10 steps.

REGISTERED BARS (frozen here before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - BLEED-KILLS-AT-ADAM-GATE: "dies (g-12 <= 0.27) with D_kill in
    [2.12, 3.27] — THE GATE REUNIFIES across trajectory classes;
    protection = staying diffusive/slow"
  - BLEED-SPARED-PAST-GATE: "passes D=2.6 alive — classes keep separate
    rings; all three named with measured kill-Ds"
  - CAP-AGAIN: "150 more steps, neither — the stall's stability is then
    the finding: a possible diffusive equilibrium that never dies;
    GRADED, no adjudication"

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * the fact = g-12 (install-60 battery mean p(Z) at ctx offset -12,
    e185's convention); KILL = g-12 <= 0.27 (the arc's SHUT bar), read
    EVERY step, interpolated linearly-in-step between the last alive read
    and the first dead read — with every-step reads the bracket is ONE
    step wide, the finest resolution this instrument has ever had.
  * displacement = cumulative ||theta_t - theta_0||_2 over all 2,739,072
    params from the ROOT (fp32, CPU, measured every step) — NOT from the
    s1200 resume point; e185/opt1/opt1b/opt1b2's currency verbatim, so D
    is comparable across all classes.
  * D_kill = the same-bracket linear interpolation of cumulative
    displacement; D = 2.6 'passed alive' = the read at the first step
    whose cumulative displacement crosses 2.6 shows g-12 > 0.27.
  * precedence (opt1b2's frozen convention, inherited): the kill wins iff
    its t_x precedes the measured 2.6-crossing step (a dead read AT the
    crossing step wins the tie); an ALIVE 2.6 crossing CERTIFIES
    BLEED-SPARED-PAST-GATE and the run CONTINUES to the cap — a kill
    AFTER a certified live crossing does NOT un-fire SPARED; its measured
    D_kill is REPORTED as the bleed's own gate (the SPARED bar's own
    letter) without re-adjudication. A kill before the crossing
    adjudicates the KILLS bar against [2.12, 3.27]; a kill OUTSIDE the
    window fires NO bar (opt1b/opt1c's convention: graded outcome, curve
    reported verbatim).
  * cap = 1350 steps TOTAL from the root (the dispatch's "150 more"),
    i.e. 150 continuation steps; CAP-AGAIN is GRADED (the stall's
    stability IS the finding) and never adjudicates.
  * reachability note (stated, not load-bearing): at the measured
    diffusive rate (~0.0008 D/step) D=2.6 sits ~570 steps out — beyond
    this cap; even a perfectly colinear 150 steps at the observed
    per-step norm (~0.0067) tops out at D ~3.2, inside the window's top.
    The arm's outcome space is exactly its three bars.

RESUME / STREAM-CONTINUITY METHOD (the dispatch's letter; opt1b2's
certified path VERBATIM, shifted to the s1200 parent): the committed s1200
state is runs/opt1b2/chunk_state.pt (saved at opt1b2's cap stop: model +
generator state + step 1200 + the 600-row continuation journal + the 68
reads + the cap stop), the exact state a cross-process resumer loads; the
final checkpoint runs/checkpoints/opt1b2_a2b_sgd_1e-2_s1200.pt holds the
same model (gated BIT-IDENTICAL here). Plain SGD(momentum 0) is STATELESS
— no optimizer state to carry; the stream position carries in the
GENERATOR STATE, so the input stream CONTINUES bit-identically from step
1201, never repeats (per-step md5s recorded; the union
opt1b(600)+opt1b2(600)+new is gated UNIQUE). The RESUME GATE (Rule 12):
the loaded model's flat md5 == opt1b2's committed hash-chain tail, the
s1200.pt model == the chunk_state model (bit), the loaded journal ==
the committed runs/opt1b2/journal.jsonl (row-equal, md5-bound) with its
last row hard-bound to the committed step-1200 trajectory row, the
committed step-1200 read hard-bound likewise, the recomputed cumulative
displacement ||theta_1200 - theta_root|| == the committed journal's
step-1200 value (bit tolerance), the committed stream's steps 1..10
x_md5s == e185's stored hashes (via opt1b's parent journal), and every
NEW step's x_md5 unique vs the whole committed+new stream.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any continuation): the
standard cell gates (corpus ZEPH count 0; splice mix {FLORIZEL: 19,
ELIZABETH: 41}; battery shapes 60 x (130 +- j); neutral bank 16 starts
bit-equal to e185's stored list; root gated vs e151's before-cells) PLUS
the resume gate above PLUS the parent's own committed verdict (opt1b2
CAP-NEITHER with kill extrapolated ~s1213.8 — the door this cell walks
through, hard-bound from the committed metrics, never recomputed).

CO-READS: (1) THE DIFFUSIVE RATE — per-step D-growth in the stall regime:
per-step walked length (step |d|) vs per-step projected growth (dD), the
sublinearity ratio, the linear and sqrt fits of D(t) over the 150 steps
(ballistic vs diffusive), g-12 stall statistics (mean/std/min/max, the
margin to the 0.27 bar, the excursion structure); (2) THE FINAL FOUR-CLASS
OVERLAY — g-12 vs D with ALL classes from COMMITTED data (opt1's four Adam
arms, opt1c's annihilation arm + its along-path densification, opt1b's
committed first-600 bleed, opt1b2's committed 601..1200 continuation —
loaded and plotted, never rerun) + this run's per-step points; (3) CE_R
every 10 steps (organism health); (4) alignment cos(delta_theta_t,
grad g0_t) and cos(delta_theta_t, grad m12_t) every 10 steps (the
critic's sign: negative = death-aligned). Monitors only — nothing
adjudicates but the three bars.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is g1bS's — never claimed), torch threads 4 (the
e185/opt1/opt1b/opt1b2 reduction order), ckpt-RESUMABLE chunks <= 180 s
per the opt1b convention (full state round-trips through
runs/opt1b3/chunk_state.pt after each chunk; each chunk LOADS the previous
chunk's file; chunk-boundary parameter md5s form a hash chain),
PROGRESSIVE PARTIAL metrics.json writes (the 2026-09-30 outage lesson,
opt1c/opt1b2's recovery convention: after the gates pass and after every
chunk save), a completed-run RECOVERY HOOK (stop persisted in chunk_state;
never steps past a completed kill), n=1, single seed lineage (10902), no
reruns beyond the cap.

Outputs: runs/opt1b3/{metrics.json, opt1b3_last_steps.png (the per-step
curve), opt1b3_four_class_overlay.png (the completed four-class overlay),
chunk_state.pt, journal.jsonl (continuation rows 1201.. only; the
committed parents' journals referenced by md5)}; checkpoint
runs/checkpoints/opt1b3_a2b_sgd_1e-2_s<final>.pt. No NOTES/THINKING/
QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python opt1b3_last_steps.py    (OPT1B3_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185/e187/opt1/opt1b/opt1b2)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185/opt1/opt1b/opt1b2-era reduction order

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("OPT1B3_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "opt1b3 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (opt1/opt1b/opt1b2's, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
OPT1B_METRICS = E43.REPO / "runs" / "opt1b" / "metrics.json"
OPT1B_JOURNAL = E43.REPO / "runs" / "opt1b" / "journal.jsonl"
OPT1B2_METRICS = E43.REPO / "runs" / "opt1b2" / "metrics.json"
OPT1B2_JOURNAL = E43.REPO / "runs" / "opt1b2" / "journal.jsonl"
OPT1B2_CHUNK = E43.REPO / "runs" / "opt1b2" / "chunk_state.pt"
OPT1B2_S1200 = CKPT_DIR / "opt1b2_a2b_sgd_1e-2_s1200.pt"
OPT1C_METRICS = E43.REPO / "runs" / "opt1c" / "metrics.json"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the arm + the continuation envelope (dispatch-frozen) -----------------------
LR_A2B = 1e-2                     # opt1 A2b / opt1b / opt1b2's lr VERBATIM
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185/opt1/opt1b/opt1b2)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
CHUNK_CAP = 178.0                 # dispatch: <=180 s per chunk (margin for step overshoot)
RESUME_STEP = 1200                # opt1b2's committed cap stop
STEP_CAP = RESUME_STEP + 150      # dispatch: 150 MORE steps (1350 total from the root)
D_SPARE = 2.6                     # the SPARED gate (opt1b/opt1b2's frozen number)
KILL_WINDOW = (2.12, 3.27)        # 2.49-2.84 +- 15% (the registered letter's window)
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE)
SURVIVE_NOTE = 0.50               # SPARE reference line (opt1's convention; report-only)
FULL_EVERY = 10                   # CE_R + g0 + alignment every 10 steps (dispatch's letter)
E170_ANCHOR_SEED = 170            # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
ADAM_GATE_BAND = (2.49, 2.84)     # opt1's measured Adam kill displacements (T139)
OPT1C_DKILL = 0.9203406595225093  # opt1c's annihilation kill displacement (overlay reference)
if SMOKE:                         # true shakedown trims (documented in deviations)
    STEP_CAP, CHUNK_CAP = RESUME_STEP + 3, 12.0

# ---- the committed s1200 resume references (hard-bound at runtime) ----------------
RESUME_MD5_1200 = "2788d4e8f2bd72f10a0a21b7d86cf00d"  # opt1b2's committed hash-chain tail
OPT1B2_JOURNAL_MD5 = "076cd8116eed4a669e09a51f1983b85a"
OPT1B_JOURNAL_MD5 = "17d1d08741204785f7f9b5825f116559"
J1200_REF = {                     # opt1b2's committed journal row at step 1200 (full precision)
    "step": 1200, "chunk": 6,
    "ce_batch": 0.6621392369270325,
    "cum_disp": 2.145502805709839,
    "step_disp": 0.006087249144911766,
    "preclip_gnorm": 0.6087709069252014,
    "lr_eff": 0.01,
    "x_md5": "79f95dd66f0ef607fe9d874a6f85b73a",
}
R1200_REF = {                     # opt1b2's committed read at step 1200 (the read anchor)
    "step": 1200, "gm12": 0.3289695680141449, "g0": 0.505268931388855,
    "ce_r": 1.6934325695037842, "cum_disp": 2.145502805709839,
}
THE_DOOR = {                      # opt1b2's committed CAP-NEITHER projection (T145's ~s1214)
    "extrapolated_kill_step": 1213.8020560539055,
    "steady_disp_rate": 0.006527118396479637,
    "extrapolated_steps_to_D_2_6": 1269.6321357576862,
}

# ---- gates / references (full precision, = stored metrics; opt1/opt1b2's set) ----
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176n's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772220270098,
    "dall_g0": 0.9047248959541321,
}
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; opt1's embedded set)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
    5: "4da7b67a7fd80b4e9729731fb27bec0c",
    6: "1aa4f9f250f14acad52d3b969343090a",
    7: "1e9e3373028935fe272d86683280e4b5",
    8: "3535a9db2d1aa2e6e0655208ff26b3d9",
    9: "2c3260242cd38e60cffafe0c91495c5e",
    10: "688062bb39f486e563b091124a7231a1",
}
D_KILL_ADAM = 2.4892616271972656  # e185/opt1-A0's kill displacement (overlay reference)

REGISTERED_PREDICTION = {
    "bleed_kills_at_adam_gate": "BLEED-KILLS-AT-ADAM-GATE: \"dies (g-12 <= "
        "0.27) with D_kill in [2.12, 3.27] — THE GATE REUNIFIES across "
        "trajectory classes; protection = staying diffusive/slow\"",
    "bleed_spared_past_gate": "BLEED-SPARED-PAST-GATE: \"passes D=2.6 alive "
        "— classes keep separate rings; all three named with measured "
        "kill-Ds\"",
    "cap_again": "CAP-AGAIN: \"150 more steps, neither — the stall's "
        "stability is then the finding: a possible diffusive equilibrium "
        "that never dies; GRADED, no adjudication\"",
    "operationalizations": "the arm = opt1b2's committed s1200 bleed state "
        "CONTINUED verbatim (plain SGD m0/wd0/clip1.0, constant lr 1e-2, "
        "the seed-10902 licensed stream whose position carries in the "
        "saved generator state; resume gated bit vs the committed "
        "chunk_state + s1200.pt + journal-tail); KILL = g-12 <= 0.27 read "
        "EVERY step, interpolated linearly-in-step between the last alive "
        "and first dead read (the every-step cadence bounds the bracket "
        "at ONE step — the finest this instrument has had); D_kill = "
        "same-bracket linear interpolation of cumulative displacement "
        "FROM THE ROOT; D = 2.6 'passed alive' = the read at the first "
        "step whose cumulative displacement crosses 2.6 shows g-12 > 0.27; "
        "stop = whichever FIRST of kill / the 1350-step cap (150 more); "
        "precedence (opt1b2's frozen convention, inherited): the kill "
        "wins iff its t_x precedes the measured 2.6-crossing step (a dead "
        "read AT the crossing wins the tie); an alive 2.6 crossing "
        "CERTIFIES SPARED and the run CONTINUES to the cap — a post-gate "
        "kill is REPORTED as the bleed's own measured kill-D (the SPARED "
        "bar's own letter), never re-adjudicated; a kill OUTSIDE "
        "[2.12, 3.27] fires NO bar (graded outcome, curve reported "
        "verbatim); composite order frozen BLEED-KILLS-AT-ADAM-GATE -> "
        "BLEED-SPARED-PAST-GATE -> CAP-AGAIN.",
    "registration": "the dispatch's registration IS the registration (the "
        "three bars quoted verbatim in the module docstring and here, "
        "frozen before compute). Adjudicate against exactly this; no bar "
        "shopping.",
    "recovery_note": "THIRD DISPATCH (second recovery): dispatch-2 "
        "executed the frozen registration and was killed mid-flight by "
        "the fifth disruption at s1264 (chunk 0: 64 every-step reads, all "
        "alive; the PARTIAL metrics.json committed at b7f8e20 + the "
        "mid-flight chunk_state.pt on disk). This run resumes from that "
        "certified chunk_state (gates.G_MIDRESUME) — it never re-executes "
        "steps 1201..1264; the bars are the dispatch's letter verbatim, "
        "unchanged.",
}

trims: list[str] = []
deviations: list[str] = [
    "THE RESUME: the run does NOT re-execute steps 1..1200 — it loads "
    "opt1b2's committed chunk_state @ s1200 (the state a cross-process "
    "resumer takes), gated bit (flat md5 == opt1b2's committed hash-chain "
    "tail; s1200.pt model bit-identical; journal row-equal to "
    "runs/opt1b2/journal.jsonl with the step-1200 trajectory row and read "
    "hard-bound; displacement recompute bit-tolerant). The continuation "
    "journal written to runs/opt1b3/journal.jsonl contains ONLY the new "
    "rows (1201..); the parents' journals are referenced by their file "
    "md5s + the embedded hard-bound row.",
    "Read cadence (the dispatch's letter): g-12 EVERY step (the kill "
    "bracket is therefore ONE step wide — the finest resolution this "
    "instrument has had); FULL read (g0 battery + CE_R + the two "
    "alignment gradients) every 10 steps, at the measured D=2.6 crossing "
    "if it occurs, and at the final step. The first continuation read "
    "anchors on the committed step-1200 read (g-12 0.3290).",
    "Chunk cap 178 s (dispatch <=180 s; the margin absorbs one step "
    "overshoot so every chunk's wall stays <= 180 s); in-chunk checkpoint "
    "reads count toward the cap (opt1's FT_TIME_CAP convention). Chunk "
    "state round-trips through runs/opt1b3/chunk_state.pt per the opt1b "
    "convention (model + generator state + step + continuation journal + "
    "reads + flags); each chunk LOADS the previous chunk's file; "
    "chunk-boundary parameter md5s form a hash chain.",
    "PROGRESSIVE PARTIAL metrics.json WRITES (the 2026-09-30 outage "
    "lesson; opt1c/opt1b2's recovery convention, mandated by this "
    "dispatch): a partial is written (a) immediately after the gates pass "
    "and (b) after every chunk save, each stamped status/phase/timestamp "
    "with gates + provenance + journal/read tails + chunks + stop-so-far; "
    "the final COMPLETE write replaces it.",
    "RESUME HARDENING (opt1c/opt1b2's recovery convention): chunk_state "
    "also persists stop + chunks_prov (appended BEFORE the save); a "
    "RECOVERY HOOK before the arm loop restores an already-completed run "
    "(stop set in chunk_state) so final metrics/plots regenerate WITHOUT "
    "stepping past a completed kill.",
    "Light per-step read = g-12 battery only (60 windows); the FULL "
    "every-10 read adds g0 + CE_R + the two fact gradients. CPU-ONLY "
    "(CUDA_VISIBLE_DEVICES=-1 before torch; the GPU is g1bS's, never "
    "claimed); threads 4; n=1; single seed lineage (10902); no reruns "
    "beyond the cap.",
    "Smoke mode trims: cap s1203 (3 continuation steps), 12 s chunks; the "
    "FULL parent resume gate still runs (that is what the shakedown "
    "exists to prove). Nothing adjudicated (verdict stamped SMOKE).",
    "Chunk walls: the chunk budget (178 s) is checked BETWEEN steps; "
    "under heavy outside CPU contention a single in-flight step can "
    "overshoot, so individual chunk walls may slightly exceed 180 s "
    "(observed max recorded in gates.G_CHUNKS). This is wall-clock "
    "contention, not compute: one training step is ~1.5-2.5 s of CPU on "
    "4 threads (opt1b/opt1b2's own rate).",
    "THIRD-DISPATCH RECOVERY (data/recovery layer only; bars untouched): "
    "the fifth disruption killed dispatch-2's executor mid-flight at s1264 "
    "(chunk 0: 64 every-step reads, all alive; survivors = the mid-flight "
    "runs/opt1b3/chunk_state.pt + the PARTIAL metrics.json committed at "
    "b7f8e20). This dispatch adds: (a) THE MID-RUN RESUME GATE "
    "(gates.G_MIDRESUME, certify_midrun) — the loaded cross-process chunk "
    "is certified before any step continues (model flat-md5 recomputed == "
    "payload == the partial's chunk-0 tail; journal contiguous from the "
    "parent's s1200 stop; one read per step; tails bit-equal the committed "
    "partial; displacement recompute from the root; stream uniqueness "
    "across opt1b+opt1b2+loaded); on failure the run ABORTS and never "
    "re-executes steps 1201..1264; (b) the chunk_idx continuity fix — a "
    "fresh process resuming a mid-flight chunk tags its chunk as "
    "st['chunk']+1 so journal rows and chunks_prov never duplicate the "
    "predecessor's chunk-0 rows (the opt1c recovery-fix class); (c) the "
    "predecessor's partial preserved verbatim under "
    "metrics['recovery']['predecessor_partial']. Verified standalone "
    "before dispatch: md5/disp recomputes exact (|d| 0.0), tails equal, "
    "union 1264-step stream unique.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/opt1b2_crossing.py VERBATIM (whose own provenance is
# lab/opt1b_sgd_kill.py via lab/opt1_optimizer_controls.py /
# lab/e185_noise_wash.py — the e176n lineage). Copied rather than imported
# to own the device policy.

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120/e151 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_x) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def flat_params(net: TinyGPT) -> torch.Tensor:
    """The fp32 flat parameter vector (all trainable tensors, net.parameters()
    order — the optimizer's own currency; 2,739,072 elements on this line)."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT READ: gradient (wrt all params, at the twin's current
    weights theta_t) of the fact battery's mean log p(Z) readout at the last
    position. The critic's sign convention: NEGATIVE cos(delta, grad) =
    displacement aligned with the DEATH gradient (W022). Consumes no RNG;
    run on the eval twin, never on the training net."""
    net.zero_grad(set_to_none=True)
    sums = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        sums.append(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
    F_obj = torch.stack(sums).sum() / ids.shape[0]
    F_obj.backward()
    g = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])
    net.zero_grad(set_to_none=True)
    return g


def flat_md5(net: TinyGPT) -> str:
    return hashlib.md5(flat_params(net).numpy().tobytes()).hexdigest()


def interp_cross(s0, v0, s1, v1, bar):
    """Linear-in-step crossing of `bar` inside the (alive -> dead) bracket."""
    return s0 + (v0 - bar) / (v0 - v1) * (s1 - s0)


def certify_midrun(st, net, theta0, j600_file, j1200_file,
                   pred_partial) -> dict:
    """THE MID-RUN RESUME GATE (third dispatch; Rule 12): certify a KILLED
    PREDECESSOR's mid-flight chunk_state BEFORE continuing from it. The
    normal per-chunk round-trip gate (flat_md5 + gen_state) already binds
    the loaded state to disk; this adds the cross-process provenance
    binds: journal contiguity from the parent's s1200 stop, one read per
    step, tails bit-equal the COMMITTED PARTIAL metrics.json (b7f8e20),
    the displacement recompute at the loaded step, and stream uniqueness
    across parents + the loaded rows. Data/recovery layer ONLY — no bar,
    cadence or adjudication logic. Failing any clause aborts the run."""
    jr, rr = st["journal"], st["reads"]
    steps = [int(r["step"]) for r in jr]
    contig = bool(steps == list(range(RESUME_STEP + 1,
                                      RESUME_STEP + 1 + len(steps))))
    reads_match = bool([int(r["step"]) for r in rr] == steps)
    md5s = ([r["x_md5"] for r in j600_file] + [r["x_md5"] for r in j1200_file]
            + [r["x_md5"] for r in jr])
    uniq = bool(len(set(md5s)) == len(md5s))
    flat = torch.cat([p.detach().reshape(-1)
                      for p in net.parameters()]).clone()
    md5_now = hashlib.md5(flat.numpy().tobytes()).hexdigest()
    disp_now = float(torch.norm(flat - theta0))
    disp_diff = abs(disp_now - float(jr[-1]["cum_disp"]))
    tail_j = tail_r = tail_c = phase_step = partial_md5 = True
    if pred_partial is not None:
        _jt = pred_partial.get("journal_tail") or []
        _rt = pred_partial.get("reads_tail") or []
        _ct = pred_partial.get("chunks") or []
        tail_j = bool(_jt == jr[-len(_jt):] if _jt else True)
        tail_r = bool(_rt == rr[-len(_rt):] if _rt else True)
        tail_c = bool(_ct == list(st.get("chunks_prov", []))[:len(_ct)]
                      if _ct else True)
        phase_step = bool(int(pred_partial.get("phase", {}).get("step", -1))
                          == int(st["step"]))
        if _ct:
            partial_md5 = bool(_ct[-1].get("flat_md5_at_end")
                               == st["flat_md5"])
    payload_md5 = bool(md5_now == st["flat_md5"])
    meta_parent_ok = "opt1b2" in str(st.get("meta", {}).get("parent", ""))
    passed = bool(payload_md5 and partial_md5 and contig and reads_match
                  and uniq and disp_diff < G_BIT_TOL and tail_j and tail_r
                  and tail_c and phase_step and meta_parent_ok)
    return {
        "ran": True, "resumed_step": int(st["step"]),
        "state_kind": ("mid-flight (stop=None) — a killed predecessor's "
                       "chunk" if st.get("stop") is None
                       else "completed (stop set)"),
        "journal_rows_loaded": len(jr), "read_rows_loaded": len(rr),
        "flat_md5_recompute_match": payload_md5,
        "partial_chunk_tail_md5_match": partial_md5,
        "journal_contiguous_from_parent_stop": contig,
        "reads_one_per_step": reads_match,
        "stream_unique_parents_plus_loaded": uniq,
        "disp_recompute_measured": disp_now,
        "disp_recompute_vs_journal_diff": disp_diff,
        "disp_recompute_bit": bool(disp_diff < G_BIT_TOL),
        "tails_equal_committed_partial": {"journal": tail_j, "reads": tail_r,
                                          "chunks": tail_c,
                                          "phase_step": phase_step},
        "meta_parent_binding_ok": meta_parent_ok,
        "pass": passed,
        "note": "THIRD-DISPATCH RECOVERY (Rule 12): the loaded cross-process "
                "chunk (written by dispatch-2's executor, killed mid-flight "
                "by the fifth disruption) is certified before ANY step "
                "continues — the model's flat-md5 recomputed in this "
                "process == the payload's recorded hash == the committed "
                "partial's chunk-0 tail; the journal is contiguous from the "
                "parent's s1200 stop with exactly one read per step; the "
                "journal/read/chunk tails are bit-equal the PARTIAL "
                "metrics.json committed at b7f8e20; the displacement from "
                "the root recomputes to the loaded journal's value; and the "
                "whole 1264-step input stream (opt1b 600 + opt1b2 600 + the "
                "loaded 64) is md5-unique. On any failure the run ABORTS "
                "(control failure) — steps 1201..1264 are never re-executed.",
    }


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("opt1b3_smoke" if SMOKE else "opt1b3")
    chunk_path = rd / "chunk_state.pt"
    # ---- the predecessor's PARTIAL (committed at b7f8e20): snapshot BEFORE
    # any write of this run overwrites runs/opt1b3/metrics.json — it rides
    # the final metrics as recovery provenance and hard-binds the mid-run
    # resume gate (certify_midrun) when this is the third dispatch.
    pred_partial = None
    if not SMOKE and (rd / "metrics.json").exists():
        try:
            _pm = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
            if _pm.get("partial"):
                pred_partial = _pm
        except (json.JSONDecodeError, OSError):
            pred_partial = None
    log("predecessor partial: "
        + (f"PRESENT (status '{pred_partial['status']}'; phase "
           f"{json.dumps(pred_partial['phase'])}) — snapshotted; mid-run "
           "resume will be CERTIFIED against it (third dispatch)"
           if pred_partial else
           "absent — this is a fresh execution of the registration"))
    log(f"OPT1B3 THE LAST FOURTEEN STEPS (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), sequential "
        f"chunks <= {CHUNK_CAP + 2:.0f}s, cap {STEP_CAP} steps total from "
        f"the root ({STEP_CAP - RESUME_STEP} more), n=1, seed lineage "
        f"{FREEZE_SEED} (A2b's)")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (OPT1_METRICS, OPT1B_METRICS, OPT1B_JOURNAL, OPT1B2_METRICS,
              OPT1B2_JOURNAL, OPT1B2_CHUNK, OPT1B2_S1200, OPT1C_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent artifact: {p}")
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    opt1bm = json.loads(OPT1B_METRICS.read_text(encoding="utf-8"))
    opt1b2m = json.loads(OPT1B2_METRICS.read_text(encoding="utf-8"))
    opt1cm = json.loads(OPT1C_METRICS.read_text(encoding="utf-8"))
    adam_arms = {t: opt1m["arms"][t]["ckpt_table"]
                 for t in ("a0_adamw_ref", "a3_adamw_warmup30",
                           "a4_adamw_b2_0.999", "a5_adamw_moment_reset")}
    bleed_s600 = opt1bm["fact_vs_D_curve"]        # the bleed to s600 (committed)
    bleed_s1200 = opt1b2m["fact_vs_D_curve_this_run"]   # opt1b2's 601..1200 reads
    anni_curve = opt1cm["fact_vs_D_curve"]        # opt1c's annihilation arm
    anni_densify = opt1cm["arm"].get("densify", [])
    chain_tail = opt1b2m["resume"]["state_hash_chain_this_run"][-1]
    parent_journal_md5 = hashlib.md5(OPT1B2_JOURNAL.read_bytes()).hexdigest()
    grandparent_journal_md5 = hashlib.md5(OPT1B_JOURNAL.read_bytes()).hexdigest()
    # hard-bind this file's embedded resume references to the COMMITTED data
    assert chain_tail == RESUME_MD5_1200, \
        "opt1b2's committed hash-chain tail drifted vs this file's copy"
    assert parent_journal_md5 == OPT1B2_JOURNAL_MD5, \
        "opt1b2's committed journal drifted vs this file's copy"
    assert grandparent_journal_md5 == OPT1B_JOURNAL_MD5, \
        "opt1b's committed journal drifted vs this file's copy"
    assert opt1b2m["adjudication"]["verdict"] == "CAP-NEITHER", \
        "the parent's committed verdict is not CAP-NEITHER"
    proj = opt1b2m["adjudication"]["bars"]["projection"]
    assert abs(float(proj["extrapolated_kill_step"])
               - THE_DOOR["extrapolated_kill_step"]) < 1e-9, \
        "the door (opt1b2's committed kill extrapolation) drifted"
    b2_parent_verdict = opt1b2m["adjudication"]["verdict"]
    log(f"parents: opt1b {opt1bm['adjudication']['verdict']}; opt1b2 "
        f"{b2_parent_verdict} (kill extrapolated s"
        f"{THE_DOOR['extrapolated_kill_step']:.1f} — THE DOOR; committed "
        f"bleed curves {len(bleed_s600)}+{len(bleed_s1200)} rows; journals "
        f"md5 {grandparent_journal_md5[:8]}…/{parent_journal_md5[:8]}…); "
        f"opt1c {opt1cm['adjudication']['verdict']} (D_kill "
        f"{OPT1C_DKILL:.4f}); none rerun")

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e185/opt1)
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
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix} (SPLICE_RNG {E43.SPLICE_RNG})")

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in GEOS},
        "expected_shapes": {"-12": [60, PRE - 12], "0": [60, PRE],
                            "12": [60, PRE + 12]},
        "pass": bool(list(bat_ids[-12].shape) == [60, PRE - 12]
                     and list(bat_ids[0].shape) == [60, PRE]
                     and list(bat_ids[12].shape) == [60, PRE + 12]),
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at ctx "
                "offsets {-12,0,+12}, e185's convention verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- the neutral stream (e170 VERBATIM via e185 arm C / opt1)
    arng = _random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                         "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — e170's "
                         "construction VERBATIM (= e185's control stream; "
                         "FIXED content, not reseeded)"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], ("neutral bank drifted vs e185's stored "
                              f"starts: {n_starts}")
    log("G_ANCHOR: neutral bank bit-matches e185's stored 16 starts: PASS")

    # ---------------- root net + gate vs e151 (opt1's gate set verbatim)
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    log(f"root: {ROOT_CK} (meta: {root_meta})")

    evl0 = copy.deepcopy(net0)
    root_cells = {
        "gm12": battery_cell(evl0, gm12_ids, zid)["mean_pz"],
        "g0": battery_cell(evl0, g0_ids, zid)["mean_pz"],
        "gp12": battery_cell(evl0, bat_ids[12], zid)["mean_pz"],
        "ce_r": ce_fixed_cpu(evl0, *r_eval_xy),
    }
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r"}
    root_refs = {keymap[k]: v for k, v in E151_ROOT.items() if k in keymap}
    rdiffs = {k: root_cells[k] - root_refs[k] for k in root_refs}
    rmax = max(abs(v) for v in rdiffs.values())
    G_ROOT = {"cells": root_cells, "refs": root_refs, "diffs": rdiffs,
              "max_abs_diff": rmax, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL, "bit": bool(rmax < G_BIT_TOL),
              "pass": bool(rmax < G_FALLBACK_TOL)}
    log(f"G_ROOT (vs e151 before-cells, light set): max|diff| {rmax:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else ""))
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED vs e151 stored "
                           "before-cells")

    theta0 = flat_params(net0)
    n_anc = anchor_neutral.shape[0]

    # =====================================================================
    # THE RESUME GATE (Rule 12) — load opt1b2's committed s1200 chunk state
    # and certify it bit: chunk_state + s1200.pt + committed journal tail.
    # =====================================================================
    pst = torch.load(OPT1B2_CHUNK, map_location="cpu", weights_only=False)
    ck1200 = torch.load(OPT1B2_S1200, map_location="cpu", weights_only=False)
    j1200 = dict(pst["journal"][-1])
    r1200 = dict(pst["reads"][-1])
    j1200_file = [json.loads(l) for l in
                  OPT1B2_JOURNAL.read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    s1200_bit = all(torch.equal(pst["model"][k], ck1200["model"][k])
                    for k in pst["model"])
    j1200_diffs = {k: abs(float(j1200[k]) - float(J1200_REF[k]))
                   for k in ("ce_batch", "cum_disp", "step_disp",
                             "preclip_gnorm")}
    r1200_diffs = {k: abs(float(r1200[k]) - float(R1200_REF[k]))
                   for k in ("gm12", "g0", "ce_r", "cum_disp")}
    journal_file_equal = bool(j1200_file == list(pst["journal"]))
    # the committed 1..10 hashes live in the GRANDPARENT (opt1b) journal
    j600_file = [json.loads(l) for l in
                 OPT1B_JOURNAL.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    committed_x_md5_1_10 = {int(r["step"]): r["x_md5"] for r in j600_file}
    xmd5_1_10_ok = all(committed_x_md5_1_10.get(s) == h
                       for s, h in E185_XHASH.items())
    parent_stop_ok = bool(pst["step"] == RESUME_STEP
                          and pst.get("stop", {}).get("kind") == "cap")
    G_RESUME = {
        "loaded_from": "runs/opt1b2/chunk_state.pt (opt1b2's committed cap "
                       "stop: model + generator state + step 1200 + 600-row "
                       "continuation journal + 68 reads + the cap stop)",
        "chunk_step": int(pst["step"]),
        "chunk_stop": E43.jsonable(pst.get("stop")),
        "chunk_meta": E43.jsonable(pst["meta"]),
        "flat_md5_loaded_vs_committed_chain_tail": bool(
            chain_tail == RESUME_MD5_1200 == pst["flat_md5"]),
        "s1200_pt_model_bit_identical": bool(s1200_bit),
        "journal_file_row_equal": journal_file_equal,
        "journal_file_md5": parent_journal_md5,
        "grandparent_journal": {"path": "runs/opt1b/journal.jsonl",
                                "md5": grandparent_journal_md5,
                                "rows": len(j600_file)},
        "step1200_row_hard_bind_diffs": j1200_diffs,
        "read1200_row_hard_bind_diffs": r1200_diffs,
        "committed_journal_x_md5_1_10_vs_e185": bool(xmd5_1_10_ok),
        "parent_committed_at_cap": parent_stop_ok,
        "max_abs_diff": max(max(j1200_diffs.values()),
                            max(r1200_diffs.values())),
        "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
        "bit": bool(max(max(j1200_diffs.values()),
                        max(r1200_diffs.values())) < G_BIT_TOL),
        "pass": bool(parent_stop_ok
                     and chain_tail == RESUME_MD5_1200 == pst["flat_md5"]
                     and s1200_bit and journal_file_equal
                     and xmd5_1_10_ok
                     and max(max(j1200_diffs.values()),
                             max(r1200_diffs.values())) < G_FALLBACK_TOL),
        "note": "PRE-DISPATCH CHECK (Rule 12): the resumed state's identity "
                "— the committed chunk_state (what a cross-process resumer "
                "loads, stopped at its own cap) vs opt1b2's committed "
                "s1200.pt model (bit), the committed journal.jsonl "
                "(row-equal) with the step-1200 trajectory row and read "
                "hard-bound to this file's embedded full-precision copies, "
                "and the committed stream's steps 1..10 x_md5s vs e185's "
                "stored hashes (via the grandparent opt1b journal). The "
                "first resumed step (1201) continues the committed "
                "trajectory bit-consistently: the state is the committed "
                "one (md5 chain) and the stream position carries in the "
                "generator state (per-step md5s recorded; never repeats — "
                "gates.G_INPUTS).",
    }
    log(f"G_RESUME (opt1b2 s1200 chunk_state + s1200.pt + journal): step "
        f"{pst['step']} stop={pst.get('stop', {}).get('kind')}, md5-chain "
        f"{'OK' if G_RESUME['flat_md5_loaded_vs_committed_chain_tail'] else 'MISMATCH'}, "
        f"s1200.pt {'BIT' if s1200_bit else 'DIFFERS'}, journal file "
        f"{'row-equal' if journal_file_equal else 'DIFFERS'}, max|d| "
        f"{G_RESUME['max_abs_diff']:.2e}: "
        + ("PASS" if G_RESUME["pass"] else "FAIL")
        + (" (bit)" if G_RESUME["bit"] else ""))
    if not G_RESUME["pass"]:
        raise RuntimeError("resume gate FAILED — the s1200 state is not "
                           "certified; abort (control failure)")

    # install the resumed state
    net = copy.deepcopy(net0)
    net.load_state_dict(pst["model"])
    net.train()
    gen = torch.Generator()
    gen.set_state(pst["gen_state"])
    evl = copy.deepcopy(net0)
    evl.eval()
    prev = flat_params(net)          # displacement continuity anchor @ s1200
    # the displacement recompute (same reduction order, this process)
    disp_recheck = float(torch.norm(prev - theta0))
    disp_recheck_diff = abs(disp_recheck - float(j1200["cum_disp"]))
    G_RESUME["disp_recompute_measured"] = disp_recheck
    G_RESUME["disp_recompute_vs_committed_diff"] = disp_recheck_diff
    G_RESUME["disp_recompute_bit"] = bool(disp_recheck_diff < G_BIT_TOL)
    if disp_recheck_diff >= G_FALLBACK_TOL:
        raise RuntimeError("displacement recompute at resume FAILED")
    log(f"G_RESUME disp recompute: {disp_recheck:.16f} vs committed "
        f"{j1200['cum_disp']:.16f} (|d| {disp_recheck_diff:.2e}, "
        f"{'bit' if disp_recheck_diff < G_BIT_TOL else 'tol'})")

    opt = torch.optim.SGD(net.parameters(), lr=LR_A2B)   # plain: m=0, wd=0
    step = int(pst["step"])
    log("WHAT THIS ARM GUARANTEES: NOTHING — it could die at ~2.2 (the "
        "gate reunifies: protection = staying diffusive/slow), pass 2.6 "
        "alive (separate rings), or stall through all 150 steps (a "
        "possible diffusive equilibrium that never dies); the openness "
        "is the point.")

    # ---- PROGRESSIVE PARTIAL WRITE #1: gates passed, continuation starting
    def write_partial(status: str, stop_: dict | None) -> None:
        save_json(rd / "metrics.json", E43.jsonable({
            "experiment": "opt1b3_last_steps", "date": common.now_iso(),
            "status": status, "partial": True,
            "phase": {"step": int(step), "chunks": len(chunks_prov),
                      "reads": len(reads), "journal_rows": len(journal),
                      "stop_kind": (stop_ or {}).get("kind"),
                      "elapsed_s": round(time.time() - T0, 1)},
            "gates_partial": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                              "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                              "G_ROOT": G_ROOT, "G_RESUME": G_RESUME,
                              "G_MIDRESUME": G_MIDRESUME},
            "provenance_partial": {
                "resume": ("continued from runs/opt1b2/chunk_state.pt "
                           f"@ step {RESUME_STEP} (gated bit: md5 chain + "
                           "s1200.pt + journal hard-bind); parent journal "
                           f"md5 {parent_journal_md5}; grandparent "
                           f"{grandparent_journal_md5}"),
                "parents": {"opt1b": opt1bm["adjudication"]["verdict"],
                            "opt1b2": b2_parent_verdict,
                            "opt1c": opt1cm["adjudication"]["verdict"]}},
            "chunks": chunks_prov,
            "journal_tail": journal[-8:], "reads_tail": reads[-8:],
            "stop_partial": stop_,
        }))

    # =====================================================================
    # THE CHUNKED CONTINUATION (1201.. to the stop) — g-12 read EVERY step
    # =====================================================================
    journal: list[dict] = []          # CONTINUATION rows only (1201..)
    reads: list[dict] = []            # continuation checkpoint reads (every step)
    chunks_prov: list[dict] = []
    zeph_checks = 0
    stop: dict | None = None
    d26_crossed = False
    d26_row: dict | None = None
    G_CHUNKS = {"round_trips": [], "pass": None}
    # the mid-run resume gate's record (filled when a CROSS-PROCESS chunk is
    # loaded in the while loop below; 'ran' stays False on a fresh start or
    # the completed-run recovery hook, whose own round-trip gate covers it)
    G_MIDRESUME: dict = {"ran": False, "pass": True,
                         "note": "not exercised (fresh start or completed-run "
                                 "recovery hook — the hook's own round-trip "
                                 "gate covers that path)"}
    chunk_idx = 0

    # ---- RECOVERY HOOK (opt1c/opt1b2's convention): a previous process
    # COMPLETED the arm (stop set in this run's chunk_state) and died before
    # the final write -> restore, skip training, regenerate outputs.
    if chunk_path.exists():
        _probe = torch.load(chunk_path, map_location="cpu",
                            weights_only=False)
        if isinstance(_probe, dict) and _probe.get("stop") is not None:
            net.load_state_dict(_probe["model"])
            net.train()
            gen.set_state(_probe["gen_state"])
            step = int(_probe["step"])
            journal = _probe["journal"]
            reads = _probe["reads"]
            d26_crossed = bool(_probe["d26_crossed"])
            d26_row = _probe["d26_row"]
            zeph_checks = int(_probe["zeph_checks"])
            chunks_prov = list(_probe.get("chunks_prov", []))
            stop = dict(_probe["stop"])
            chunk_idx = int(_probe["chunk"]) + 1
            prev = flat_params(net)
            _rt = {"chunk": int(_probe["chunk"]), "loaded_step": step,
                   "flat_md5_match": bool(flat_md5(net)
                                          == _probe["flat_md5"]),
                   "gen_state_match": bool(torch.equal(
                       gen.get_state(), _probe["gen_state"])),
                   "journal_len": len(journal), "recovery_hook": True}
            G_CHUNKS["round_trips"].append(_rt)
            log(f"RECOVERY HOOK: completed run restored @ step {step} "
                f"(stop={stop['kind']}; md5 "
                f"{'OK' if _rt['flat_md5_match'] else 'MISMATCH'}, gen "
                f"{'OK' if _rt['gen_state_match'] else 'MISMATCH'}) — "
                f"NO re-training")
            if not (_rt["flat_md5_match"] and _rt["gen_state_match"]):
                raise RuntimeError("recovery hook round-trip FAILED")
            write_partial(
                f"PARTIAL (recovery hook): completed run restored @ "
                f"step {step} (stop={stop['kind']}); final write pending",
                stop)

    if stop is None:
        write_partial("PARTIAL — all pre-dispatch gates PASSED (resume "
                      "certified bit; continuation starting; no steps yet)",
                      None)

    while stop is None:
        # ---- chunk begin: LOAD the previous chunk's state file (the
        # cross-process resume path; chunk 0 starts from the parent state)
        if chunk_path.exists():
            st = torch.load(chunk_path, map_location="cpu",
                            weights_only=False)
            net.load_state_dict(st["model"])
            net.train()
            gen.set_state(st["gen_state"])
            step = int(st["step"])
            journal = st["journal"]
            reads = st["reads"]
            d26_crossed = bool(st["d26_crossed"])
            d26_row = st["d26_row"]
            # assign, never accumulate (opt1c's recovery fix)
            zeph_checks = int(st["zeph_checks"])
            chunks_prov = list(st.get("chunks_prov", []))
            # cross-process chunk continuity: this process's chunk is the
            # NEXT one (a fresh process resuming a mid-flight chunk 0 must
            # not re-tag its rows/chunk record as chunk 0 — the opt1c
            # recovery-fix class; a no-op for in-process boundaries)
            chunk_idx = int(st["chunk"]) + 1
            prev = flat_params(net)          # displacement continuity anchor
            rt = {"chunk": int(st["chunk"]) + 1, "loaded_step": step,
                  "flat_md5_match": bool(flat_md5(net) == st["flat_md5"]),
                  "gen_state_match": bool(torch.equal(
                      gen.get_state(), st["gen_state"])),
                  "journal_len": len(journal)}
            G_CHUNKS["round_trips"].append(rt)
            log(f"[chunk {chunk_idx}] LOADED chunk_state.pt @ step {step} "
                f"(md5 {'OK' if rt['flat_md5_match'] else 'MISMATCH'}, "
                f"gen {'OK' if rt['gen_state_match'] else 'MISMATCH'})")
            if not (rt["flat_md5_match"] and rt["gen_state_match"]):
                raise RuntimeError("chunk round-trip FAILED — state not "
                                   "resumable")
            # ---- THE MID-RUN RESUME GATE (third dispatch; Rule 12): the
            # loaded chunk may be a KILLED PREDECESSOR's mid-flight state —
            # certify it (provenance binds, above) before ANY step
            # continues; a normal in-process chunk boundary passes the same
            # certification trivially (its rows were written by this process).
            G_MIDRESUME.clear()
            G_MIDRESUME.update(certify_midrun(st, net, theta0, j600_file,
                                              j1200_file, pred_partial))
            log(f"G_MIDRESUME ({G_MIDRESUME['state_kind']}): resumed @ s"
                f"{G_MIDRESUME['resumed_step']}, {G_MIDRESUME['journal_rows_loaded']} "
                f"journal rows, md5 recompute "
                f"{'OK' if G_MIDRESUME['flat_md5_recompute_match'] else 'MISMATCH'}, "
                f"disp recompute |d| "
                f"{G_MIDRESUME['disp_recompute_vs_journal_diff']:.2e}, tails-vs-"
                f"committed-partial {G_MIDRESUME['tails_equal_committed_partial']}: "
                + ("PASS" if G_MIDRESUME["pass"] else "FAIL"))
            if not G_MIDRESUME["pass"]:
                raise RuntimeError(
                    "MID-RUN RESUME GATE FAILED — the predecessor's "
                    f"chunk_state @ s{st['step']} is not certified; abort "
                    "(control failure)")
        else:
            log(f"[chunk {chunk_idx}] fresh start from the parent's "
                f"committed s1200 state (step {step})")

        t_chunk = time.time()

        # ---- one chunk: steps until stop / cap / chunk-budget
        while stop is None:
            if step >= STEP_CAP:
                stop = {"kind": "cap", "step": step,
                        "note": "150-more-step cap reached (1350 total) "
                                "with neither the kill nor a live 2.6 "
                                "crossing inside the window"}
                break
            if (time.time() - t_chunk) > CHUNK_CAP:
                log(f"[chunk {chunk_idx}] chunk budget "
                    f"{CHUNK_CAP:.0f}s at step {step}")
                break
            step += 1
            aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                               generator=gen)
            anc = anchor_neutral[aj]
            rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
            for w in rnd:                      # name-free VERIFY (hard-fail)
                txt = "".join(itos[int(c)] for c in w[:64]) + \
                      "".join(itos[int(c)] for c in w[192:])
                if "ZEPH" in txt:
                    zeph_checks += 1
            x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
            y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
            xh = hashlib.md5(x.contiguous().numpy().tobytes()).hexdigest()
            for grp in opt.param_groups:
                grp["lr"] = LR_A2B
            logits, _ = net(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                         1.0))
            opt.step()
            cur = flat_params(net)
            cum_disp = float(torch.norm(cur - theta0))
            inc_disp = float(torch.norm(cur - prev))
            prev = cur
            journal.append({"step": step, "chunk": chunk_idx,
                            "ce_batch": float(loss.item()),
                            "cum_disp": cum_disp, "step_disp": inc_disp,
                            "preclip_gnorm": gnorm, "lr_eff": LR_A2B,
                            "x_md5": xh})

            # ---- EVERY-STEP read (the dispatch's letter): g-12 ALWAYS;
            # the FULL read (g0 + CE_R + alignment) every 10 steps, at the
            # D=2.6 crossing, and at the cap.
            forced_flags = []
            if (not d26_crossed) and cum_disp >= D_SPARE:
                d26_crossed = True
                forced_flags.append("d26")
            forced = "+".join(forced_flags) if forced_flags else None
            at_cap = step >= STEP_CAP
            full = bool((step % FULL_EVERY == 0) or forced_flags or at_cap)
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            evl.load_state_dict(sd_cpu)
            evl.eval()
            delta = cur - theta0
            gz = battery_cell(evl, gm12_ids, zid)
            rrow = {"step": step, "chunk": chunk_idx, "forced": forced,
                    "full": full,
                    "gm12": gz["mean_pz"],
                    "g0": None, "ce_r": None,
                    "frac_argmax_z": gz["frac_argmax_z"],
                    "cum_disp": cum_disp,
                    "cos_delta_fact_g0": None, "cos_delta_fact_m12": None}
            if full:
                gz0 = battery_cell(evl, g0_ids, zid)
                ce_r = ce_fixed_cpu(evl, *r_eval_xy)
                g_g0 = fact_grad(evl, g0_ids, zid)
                g_m12 = fact_grad(evl, gm12_ids, zid)
                rrow["g0"] = gz0["mean_pz"]
                rrow["ce_r"] = ce_r
                rrow["cos_delta_fact_g0"] = float(
                    torch.dot(delta, g_g0)
                    / (torch.norm(delta) * torch.norm(g_g0) + 1e-30))
                rrow["cos_delta_fact_m12"] = float(
                    torch.dot(delta, g_m12)
                    / (torch.norm(delta) * torch.norm(g_m12) + 1e-30))
            reads.append(rrow)
            if "d26" in forced_flags:
                d26_row = dict(rrow)
            log(f"  +{step:4d} g-12 {gz['mean_pz']:.4f}"
                + (f" g0 {rrow['g0']:.4f} CE_R {rrow['ce_r']:.4f}"
                   f" cos(g0) {rrow['cos_delta_fact_g0']:+.3f}" if full
                   else "")
                + f" |d| {cum_disp:.5f}"
                + (f"  <= D={D_SPARE} FORCED" if "d26" in forced_flags
                   else ""))
            # ---- stop checks (kill first at a tie step; composite order)
            if gz["mean_pz"] <= SHUT_BAR:
                prevr = reads[-2] if len(reads) >= 2 else {
                    "step": RESUME_STEP, "gm12": float(r1200["gm12"]),
                    "cum_disp": float(r1200["cum_disp"])}
                s0, s1 = prevr["step"], step
                v0, v1 = prevr["gm12"], gz["mean_pz"]
                d0, d1 = prevr["cum_disp"], cum_disp
                t_x = interp_cross(s0, v0, s1, v1, SHUT_BAR)
                d_kill = d0 + (t_x - s0) / (s1 - s0) * (d1 - d0)
                stop = {"kind": "kill", "step": step, "t_x": t_x,
                        "bracket": (s0, s1),
                        "gm12_bracket": (v0, v1),
                        "D_bracket": (d0, d1),
                        "D_kill_interp": d_kill,
                        "bracket_width_steps": s1 - s0,
                        "d26_row": (dict(d26_row) if d26_row else None),
                        "post_gate": bool(d26_row is not None
                                          and d26_row["gm12"] > SHUT_BAR)}
                break

        # ---- chunk end: SAVE the full state (round-trip through disk).
        # chunks_prov appended BEFORE the save; stop persisted (recovery).
        fp_md5 = flat_md5(net)
        ch_rows = [r["step"] for r in journal if r["chunk"] == chunk_idx]
        chunks_prov.append({"chunk": chunk_idx,
                            "step_from": (min(ch_rows) if ch_rows
                                          else step + 1),
                            "step_to": step,
                            "n_steps": sum(1 for r in journal
                                           if r["chunk"] == chunk_idx),
                            "wall_s": round(time.time() - t_chunk, 1),
                            "flat_md5_at_end": fp_md5,
                            "stopped": stop is not None})
        payload = {"model": {k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items()},
                   "gen_state": gen.get_state().clone(),
                   "step": int(step), "chunk": int(chunk_idx),
                   "journal": journal, "reads": reads,
                   "d26_crossed": bool(d26_crossed),
                   "d26_row": d26_row,
                   "zeph_checks": int(zeph_checks),
                   "stop": stop,
                   "chunks_prov": list(chunks_prov),
                   "flat_md5": fp_md5,
                   "meta": {"experiment": "opt1b3", "arm": "a2b_sgd_1e-2",
                            "lr": LR_A2B, "input_seed": FREEZE_SEED,
                            "base": f"runs/checkpoints/{ROOT_CK}",
                            "parent": "runs/opt1b2/chunk_state.pt @ s1200 "
                                      "(committed; journal md5 "
                                      f"{parent_journal_md5})"}}
        torch.save(payload, chunk_path)
        write_partial(
            f"PARTIAL: chunk {chunk_idx} saved @ step {step} "
            f"(stop={stop['kind'] if stop else 'none'}); "
            + ("final write pending" if stop else "continuation continuing"),
            stop)
        log(f"[chunk {chunk_idx}] SAVED @ step {step} "
            f"(wall {time.time() - t_chunk:.0f}s, md5 {fp_md5[:10]}…)")
        chunk_idx += 1
        if stop is not None:
            break
        if step >= STEP_CAP:
            stop = {"kind": "cap", "step": step,
                    "note": "150-more-step cap reached (1350 total) "
                            "with neither the kill nor a live 2.6 "
                            "crossing inside the window"}
            break

    G_CHUNKS["n_chunks"] = chunk_idx
    G_CHUNKS["pass"] = bool(all(rt["flat_md5_match"]
                                and rt["gen_state_match"]
                                for rt in G_CHUNKS["round_trips"]))
    G_CHUNKS["max_chunk_wall_s"] = max((c["wall_s"] for c in chunks_prov),
                                       default=0.0)
    G_DRAWFREE = {"zeph_violations": zeph_checks,
                  "pass": bool(zeph_checks == 0)}
    assert G_DRAWFREE["pass"], "name token leaked into a window"
    # ---- the input-stream gate for a CONTINUATION: the committed stream's
    # steps 1..10 re-certified (G_RESUME), the NEW stream recorded and
    # UNIQUE across committed+new (never repeats; battery reads no RNG).
    committed_md5s = ([r["x_md5"] for r in j600_file]
                      + [r["x_md5"] for r in j1200_file])
    new_md5s = [r["x_md5"] for r in journal]
    uniq_ok = (len(set(committed_md5s + new_md5s))
               == len(committed_md5s + new_md5s))
    G_INPUTS = {
        "committed_journal_stream": {"rows": len(committed_md5s),
                                     "steps_1_10_vs_e185": bool(xmd5_1_10_ok)},
        "new_rows": len(new_md5s),
        "all_md5s_unique_committed_plus_new": bool(uniq_ok),
        "pass": bool(xmd5_1_10_ok and uniq_ok and len(new_md5s) > 0),
        "note": "the continuation's input gate: the committed stream is "
                "certified in G_RESUME (steps 1..10 vs e185's stored "
                "hashes + the row-equal journals); every NEW step's x_md5 "
                "is recorded in the journal and the union committed+new "
                "is UNIQUE — the resumed generator state continues the "
                "stream bit-consistently and never repeats",
    }
    assert G_INPUTS["pass"], "input stream failed the continuation gate"
    log(f"G_INPUTS: {len(new_md5s)} new steps, all md5s unique across "
        f"committed+new, committed 1..10 re-certified: PASS")
    log(f"G_CHUNKS: {chunk_idx} chunks, {len(G_CHUNKS['round_trips'])} "
        f"disk round-trips, max chunk wall "
        f"{G_CHUNKS['max_chunk_wall_s']:.0f}s: "
        + ("PASS" if G_CHUNKS["pass"] else "FAIL"))

    # ---- the final checkpoint (provenance for any follow-up cell)
    fin_step = step
    fin_name = ("smoke_" if SMOKE else "") + \
        f"opt1b3_a2b_sgd_1e-2_s{fin_step}"
    torch.save({"model": {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()},
                "meta": {"experiment": "opt1b3", "arm": "a2b_sgd_1e-2",
                         "steps": int(fin_step), "lr": LR_A2B,
                         "input_seed": FREEZE_SEED,
                         "stop": stop["kind"],
                         "base": f"runs/checkpoints/{ROOT_CK}",
                         "parent": "runs/opt1b2/chunk_state.pt @ s1200 "
                                   "+ runs/opt1b3/chunk_state.pt"}},
               CKPT_DIR / f"{fin_name}.pt")
    log(f"[ckpt] saved {fin_name}.pt (stop={stop['kind']} @ s{fin_step})")

    # =====================================================================
    # THE CO-READ: THE DIFFUSIVE RATE in the stall regime (monitor only)
    # =====================================================================
    diffusive = None
    if len(journal) >= 2:
        dd = np.diff(np.array([r["cum_disp"] for r in journal]))
        sd = np.array([r["step_disp"] for r in journal])[1:]
        x = np.array([r["step"] - RESUME_STEP for r in journal],
                     dtype=float)
        y = np.array([r["cum_disp"] for r in journal]) - float(
            j1200["cum_disp"])
        # linear fit D - D0 = a * (t - t0)
        A = np.vstack([x, np.ones_like(x)]).T
        (a_lin, b_lin), res_lin, *_ = np.linalg.lstsq(
            A, y, rcond=None)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2_lin = (1.0 - float(res_lin[0]) / ss_tot
                  if len(res_lin) and ss_tot > 0 else None)
        # sqrt fit D - D0 = b * sqrt(t - t0) (the diffusive signature)
        sq = np.sqrt(np.maximum(x, 0.0))
        B = np.vstack([sq, np.ones_like(sq)]).T
        (a_sq, b_sq), res_sq, *_ = np.linalg.lstsq(
            B, y, rcond=None)
        r2_sq = (1.0 - float(res_sq[0]) / ss_tot
                 if len(res_sq) and ss_tot > 0 else None)
        g = np.array([r["gm12"] for r in reads])
        gs = np.array([r["step"] for r in reads], dtype=float)
        g_slope = float(np.polyfit(gs - gs.mean(), g - g.mean(), 1)[0]) \
            if len(g) >= 2 else None
        diffusive = {
            "mean_step_disp": float(sd.mean()),
            "mean_delta_D": float(dd.mean()),
            "std_delta_D": float(dd.std()),
            "sublinearity_ratio_mean_step_over_deltaD": float(
                sd.mean() / dd.mean()) if dd.mean() > 0 else None,
            "D_traveled_this_run": float(y[-1]),
            "linear_fit": {"slope_D_per_step": float(a_lin),
                           "intercept": float(b_lin), "r2": r2_lin},
            "sqrt_fit": {"slope_D_per_sqrt_step": float(a_sq),
                         "intercept": float(b_sq), "r2": r2_sq},
            "ballistic_reference_D_per_step": float(sd.mean()),
            "note": ("THE DIFFUSIVE RATE (co-read): per-step walked length "
                     "vs projected growth; sqrt beats linear = the walk "
                     "stays diffusive; linear at mean step_disp = it went "
                     "ballistic (colinear steps). T145's stall read "
                     "~0.0008 D/step against a walked ~0.0065."),
        }
        stall = {
            "n_step_reads": len(reads),
            "gm12_mean": float(g.mean()), "gm12_std": float(g.std()),
            "gm12_min": float(g.min()), "gm12_max": float(g.max()),
            "gm12_first": float(g[0]), "gm12_last": float(g[-1]),
            "min_margin_to_shut_bar": float(g.min() - SHUT_BAR),
            "steps_below_0p30": int((g < 0.30).sum()),
            "gm12_slope_per_step": g_slope,
            "note": ("THE STALL'S STABILITY (CAP-AGAIN's own finding): "
                     "per-step g-12 over the whole continuation — the "
                     "excursion structure around ~0.33, the margin to "
                     "the 0.27 DISSOLVE bar, and the trend slope. If "
                     "neither bar fired, THIS is the graded result: a "
                     "possible diffusive equilibrium that never dies."),
        }
    else:
        stall = None

    # =====================================================================
    # ADJUDICATION (registered clauses; composite order frozen:
    # BLEED-KILLS-AT-ADAM-GATE -> BLEED-SPARED-PAST-GATE -> CAP-AGAIN;
    # precedence inherited from opt1b2: the kill wins iff its t_x precedes
    # the measured 2.6-crossing step; kill first at a tie step; no shopping)
    # =====================================================================
    kill_window_lo, kill_window_hi = KILL_WINDOW
    bars: dict = {}
    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "shakedown only — the resume path was the point"
    elif stop["kind"] == "kill" and not stop.get("post_gate"):
        # the kill's t_x precedes the 2.6 crossing (or the crossing never
        # happened): the KILLS bar adjudicates on D_kill vs the window
        d_kill = stop["D_kill_interp"]
        in_window = bool(kill_window_lo <= d_kill <= kill_window_hi)
        bars = {
            "BLEED_KILLS_AT_ADAM_GATE": {"fires": in_window,
                                         "D_kill_interp": d_kill,
                                         "window": list(KILL_WINDOW),
                                         "in_window": in_window,
                                         "t_x": stop["t_x"],
                                         "bracket": list(stop["bracket"]),
                                         "bracket_width_steps":
                                             stop["bracket_width_steps"],
                                         "gm12_bracket": list(
                                             stop["gm12_bracket"]),
                                         "D_bracket": list(stop["D_bracket"])},
            "BLEED_SPARED_PAST_GATE": {"fires": False},
            "CAP_AGAIN": {"fires": False},
        }
        if in_window:
            verdict = "BLEED-KILLS-AT-ADAM-GATE"
            clause = (f"the bleed died (g-12 <= {SHUT_BAR}) at "
                      f"interpolated displacement D_kill {d_kill:.4f} "
                      f"(t_x {stop['t_x']:.2f}, one-step bracket "
                      f"{stop['bracket']}, g-12 bracket "
                      f"{stop['gm12_bracket']}) — INSIDE the registered "
                      f"window [{kill_window_lo}, {kill_window_hi}]: THE "
                      f"GATE REUNIFIES across trajectory classes; "
                      f"protection = staying diffusive/slow. The "
                      f"last-slope door (opt1b2's extrapolated kill "
                      f"~s{THE_DOOR['extrapolated_kill_step']:.1f}) is "
                      f"walked through and measured.")
        else:
            verdict = "KILL-OUT-OF-WINDOW (no bar)"
            clause = (f"the bleed died at interpolated displacement D_kill "
                      f"{d_kill:.4f} (t_x {stop['t_x']:.2f}) — OUTSIDE the "
                      f"registered window [{kill_window_lo}, "
                      f"{kill_window_hi}]: NEITHER bar fires (the graded "
                      f"convention: the full per-step curve is reported "
                      f"verbatim; no bar claim).")
    elif stop["kind"] == "kill" and stop.get("post_gate"):
        # the 2.6 crossing was passed ALIVE first (whichever-first in time):
        # SPARED stands; the post-gate kill is REPORTED as the bleed's own
        # measured gate (the SPARED bar's own letter), never re-adjudicated
        d26 = stop.get("d26_row") or {}
        bars = {
            "BLEED_KILLS_AT_ADAM_GATE": {"fires": False},
            "BLEED_SPARED_PAST_GATE": {
                "fires": True,
                "d26_row": d26,
                "post_gate_kill": {
                    "D_kill_interp": stop["D_kill_interp"],
                    "t_x": stop["t_x"],
                    "bracket": list(stop["bracket"]),
                    "gm12_bracket": list(stop["gm12_bracket"]),
                    "D_bracket": list(stop["D_bracket"]),
                    "note": "the bleed's OWN measured kill-D (past the "
                            "gate; reported per the SPARED bar's letter, "
                            "not adjudicated against the Adam window)"}},
            "CAP_AGAIN": {"fires": False},
        }
        verdict = "BLEED-SPARED-PAST-GATE (post-gate kill reported)"
        clause = (f"the bleed PASSED D = 2.6 alive (g-12 "
                  f"{d26.get('gm12', float('nan')):.4f} > {SHUT_BAR} at D "
                  f"{d26.get('cum_disp', float('nan')):.4f}, step "
                  f"{d26.get('step', '?')}) — classes keep separate rings; "
                  f"all three named with measured kill-Ds (guillotine "
                  f"{D_KILL_ADAM:.3f} / annihilation {OPT1C_DKILL:.3f} / "
                  f"bleed its own: measured kill-D "
                  f"{stop['D_kill_interp']:.4f} at t_x {stop['t_x']:.2f}, "
                  f"bracket {stop['bracket']}).")
    elif stop["kind"] == "cap" and d26_row is not None:
        # the crossing happened alive; the run continued to the cap per the
        # stop rule (whichever-first: the crossing CERTIFIED SPARED in time)
        d26 = dict(d26_row)
        bars = {
            "BLEED_KILLS_AT_ADAM_GATE": {"fires": False},
            "BLEED_SPARED_PAST_GATE": {"fires": True,
                                       "d26_row": d26},
            "CAP_AGAIN": {"fires": False},
        }
        verdict = "BLEED-SPARED-PAST-GATE"
        clause = (f"the bleed PASSED D = 2.6 alive (g-12 "
                  f"{d26.get('gm12', float('nan')):.4f} at D "
                  f"{d26.get('cum_disp', float('nan')):.4f}, step "
                  f"{d26.get('step', '?')}) and stayed alive to the cap "
                  f"(s{step}, g-12 {reads[-1]['gm12']:.4f}) — classes keep "
                  f"separate rings; the bleed's own kill-D remains "
                  f"unmeasured past the cap (EXTRAPOLATED only, never "
                  f"adjudicated).")
    else:
        # CAP-AGAIN: neither — the stall's stability IS the finding
        bars = {
            "BLEED_KILLS_AT_ADAM_GATE": {"fires": False},
            "BLEED_SPARED_PAST_GATE": {"fires": False},
            "CAP_AGAIN": {"fires": True},
        }
        verdict = "CAP-AGAIN"
        clause = (f"150 more steps, neither: no step read g-12 <= "
                  f"{SHUT_BAR} and D = 2.6 was never crossed (final D "
                  f"{journal[-1]['cum_disp']:.4f} at s{step}, g-12 "
                  f"{reads[-1]['gm12']:.4f}) — the stall's stability is "
                  f"then the finding: a possible diffusive equilibrium "
                  f"that never dies; GRADED, no adjudication.")
    log("=" * 78)
    log(f"OPT1B3 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # ---- alignment summary (co-read; report, no adjudication)
    full_reads = [r for r in reads if r["full"]]
    align_summary = None
    if full_reads:
        align_summary = {
            "mean_cos_g0": (sum(r["cos_delta_fact_g0"] for r in full_reads)
                            / len(full_reads)),
            "mean_cos_m12": (sum(r["cos_delta_fact_m12"] for r in full_reads)
                             / len(full_reads)),
            "min_cos_g0": min(r["cos_delta_fact_g0"] for r in full_reads),
            "max_cos_g0": max(r["cos_delta_fact_g0"] for r in full_reads),
            "note": "the W022/W023 question in the stall: opt1b2's late "
                    "window sat at cos(g0) ~-0.020 (flat, RAW-WINS-"
                    "consistent); does the stall drift death-directed "
                    "(-) or relax (0)?",
        }

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "opt1b3_last_steps",
        "date": common.now_iso(),
        "status": "COMPLETE — adjudicated (this write replaces all PARTIAL "
                  "progressive writes)",
        "registration": ("the dispatch's registration IS the registration "
                         "(the three bars quoted verbatim in the module "
                         "docstring and in registered_prediction, frozen "
                         "before compute)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("THE LAST FOURTEEN STEPS: opt1b2's diffusive bleed "
                     "stalled at g-12 0.3290 / D 2.1455 at the 1200-step "
                     "cap, INSIDE the kill window's lower margin, alive; "
                     "the last-slope extrapolation reads the kill ~s1214. "
                     "A tens-of-steps continuation decides: the bleed's "
                     "OWN kill-D."),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "e185 arm C (CONTROL) / opt1 A2b / opt1b / "
                             "opt1b2 VERBATIM — the consolidated host fact "
                             "under its neutral/corpus wash",
            "batch": f"32 = {ANCH_BS} neutral-bank anchors (seed "
                     f"{E170_ANCHOR_SEED}, fixed) + {RAND_BS} random "
                     "corpus windows, full-token CE, clip 1.0",
            "targets": "TRUE (the real neutral wash — the continuation "
                       "changes NOTHING vs the committed bleed: same "
                       "optimizer, same stream, same clip; it is the same "
                       "training, resumed)",
            "input_seed": FREEZE_SEED,
            "opt_spec": {"kind": "SGD", "lr": LR_A2B, "momentum": 0,
                         "weight_decay": 0.0, "clip": 1.0,
                         "lr_schedule": "constant"},
            "read_cadence": {"g12": "EVERY STEP (the kill bracket is ONE "
                                    "step wide)",
                             "full_every": FULL_EVERY,
                             "full_contents": "g0 battery + CE_R + the two "
                                              "alignment gradients",
                             "forced_reads": ["D=2.6 crossing (if reached)",
                                              "cap"]},
            "measure_light": "g-12 EVERY step + D(t) per step (journal); "
                             "CE_R + g0 + alignment every 10 steps (the "
                             "dispatch's read list; no census/deletions)",
        },
        "resume": {
            "method": ("CONTINUE from opt1b2's committed s1200 state: "
                       "runs/opt1b2/chunk_state.pt (model + generator "
                       "state + step 1200 + the 600-row continuation "
                       "journal + the 68 reads + the cap stop — the state "
                       "a cross-process resumer loads), cross-gated bit vs "
                       "runs/checkpoints/opt1b2_a2b_sgd_1e-2_s1200.pt; "
                       "plain SGD m=0/wd=0 is STATELESS (no optimizer "
                       "state to carry); the stream position carries in "
                       "the generator state — the input stream CONTINUES "
                       "bit-identically from step 1201, never repeats "
                       "(union md5 uniqueness gated across "
                       "opt1b(600)+opt1b2(600)+new)"),
            "parent_journals": {"opt1b2": {"path": "runs/opt1b2/journal.jsonl",
                                           "rows": len(j1200_file),
                                           "md5": parent_journal_md5},
                                "opt1b": {"path": "runs/opt1b/journal.jsonl",
                                          "rows": len(j600_file),
                                          "md5": grandparent_journal_md5}},
            "parent_state_hash_chain_tail": chain_tail,
            "state_hash_chain_this_run": [c["flat_md5_at_end"]
                                          for c in chunks_prov],
            "the_door": THE_DOOR,
            "gates": "see gates.G_RESUME (the resume bit-consistency "
                     "gate: md5 chain + s1200.pt bit + journal row-equal "
                     "+ hard-bound step-1200 row/read + displacement "
                     "recompute + committed 1..10 x_md5s + the parent's "
                     "own cap stop)",
        },
        "arm": {
            "tag": "a2b_sgd_1e-2 (continued past the s1200 cap)",
            "parent": "runs/opt1b2/metrics.json arms (1200 steps, "
                      "CAP-NEITHER, alive 0.3290 at D 2.1455; kill "
                      "extrapolated s1213.8 — EXTRAPOLATED, never "
                      "adjudicated)",
            "max_steps": STEP_CAP, "steps_ran": fin_step,
            "resume_step": RESUME_STEP,
            "continuation_steps": fin_step - RESUME_STEP,
            "train_seconds": round(sum(c["wall_s"] for c in chunks_prov),
                                   1),
            "time_cap_per_chunk": CHUNK_CAP,
            "seed": FREEZE_SEED,
            "traj": journal, "ckpt_table": reads,
            "zeph_violations": zeph_checks,
        },
        "chunks": chunks_prov,
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT, "G_RESUME": G_RESUME,
                  "G_MIDRESUME": G_MIDRESUME,
                  "G_INPUTS": G_INPUTS, "G_DRAWFREE": G_DRAWFREE,
                  "G_CHUNKS": G_CHUNKS},
        "recovery": {
            "dispatch": 3,
            "note": ("THIRD dispatch (second recovery): dispatch-2 executed "
                     "the frozen registration and was killed mid-flight by "
                     "the fifth disruption (user-confirmed system shutdown, "
                     "2026-10-01) at s1264, after chunk 0's 64 every-step "
                     "reads (all alive, g-12 0.316-0.342, D 2.1918). This "
                     "run resumed from that state — certified by "
                     "gates.G_MIDRESUME (model md5 recompute; journal "
                     "contiguity; one read per step; tails bit-equal the "
                     "committed partial; displacement recompute; stream "
                     "uniqueness) — and NEVER re-executed steps 1201..1264. "
                     "The predecessor's PARTIAL metrics.json (progressive "
                     "write, committed at b7f8e20) is preserved VERBATIM "
                     "below. No bar, cadence, gate threshold or adjudication "
                     "logic changed by the recovery."),
            "predecessor_partial_committed_at": "b7f8e20",
            "predecessor_partial": pred_partial,
        },
        "references": {
            "opt1": {"metrics": "runs/opt1/metrics.json",
                     "role": "the Adam arms' fact-vs-D curves (the "
                             "overlay, plotted from committed data, never "
                             "rerun); D_kill 2.489 and the gate band "
                             "2.49-2.84 (T139)"},
            "opt1b": {"metrics": "runs/opt1b/metrics.json",
                      "role": "the bleed's committed first 600 steps "
                              "(T142's CAP-NEITHER; the grandparent "
                              "journal re-certifies steps 1..10 vs e185)"},
            "opt1b2": {"metrics": "runs/opt1b2/metrics.json",
                       "role": "the committed s600->s1200 continuation "
                               "(T145's CAP-NEITHER + THE DIFFUSIVE WALK "
                               "falsification); the resume point "
                               "(chunk_state @ s1200) and the door "
                               "(extrapolated kill ~s1213.8)"},
            "opt1c": {"metrics": "runs/opt1c/metrics.json",
                      "role": "the annihilation arm (raw-gradient "
                              "direction at Adam's size: D_kill 0.920; "
                              "its curve + along-path densification are "
                              "the overlay's second class); T143's "
                              "KILL-OUT-OF-WINDOW"},
            "e185": {"metrics": "runs/e185/metrics.json",
                     "role": "the licensed cell + the stored input md5s "
                             "the committed stream re-derives at steps "
                             "1..10"},
        },
        "fact_vs_D_curve_this_run": [{"step": r["step"], "D": r["cum_disp"],
                                      "gm12": r["gm12"], "g0": r["g0"],
                                      "ce_r": r["ce_r"],
                                      "frac_argmax_z": r["frac_argmax_z"],
                                      "cos_delta_fact_g0":
                                          r["cos_delta_fact_g0"],
                                      "cos_delta_fact_m12":
                                          r["cos_delta_fact_m12"],
                                      "forced": r["forced"],
                                      "full": r["full"]}
                                     for r in reads],
        "bleed_committed_overlay": {
            "opt1b_to_s600": [{"step": r["step"], "D": r["D"],
                               "gm12": r["gm12"]}
                              for r in bleed_s600],
            "opt1b2_to_s1200": [{"step": r["step"], "D": r["D"],
                                 "gm12": r["gm12"]}
                                for r in bleed_s1200]},
        "adam_overlay": {t: [{"step": r["step"], "D": r["cum_disp"],
                              "gm12": r["gm12"]}
                             for r in rows if r["step"] > 0]
                         for t, rows in adam_arms.items()},
        "annihilation_overlay": {"curve": [{"step": r["step"],
                                            "D": r["D"], "gm12": r["gm12"]}
                                           for r in anni_curve],
                                 "densify": anni_densify,
                                 "D_kill": OPT1C_DKILL},
        "coreads": {
            "diffusive_rate": diffusive,
            "stall_stability": stall,
            "alignment": align_summary,
            "ce_r": {"note": "does the organism keep learning in the "
                             "stall? CE_R every 10 steps (rows in "
                             "fact_vs_D_curve_this_run)",
                     "root": root_cells["ce_r"],
                     "s1200_committed": r1200["ce_r"],
                     "final": reads[-1]["ce_r"] if reads else None},
        },
        "adjudication": {
            "bars": bars,
            "verdict": verdict, "clause": clause,
            "stop": {k: v for k, v in stop.items()},
            "constants": {"D_spare_gate": D_SPARE,
                          "kill_window": list(KILL_WINDOW),
                          "adam_gate_band": list(ADAM_GATE_BAND),
                          "shut_bar": SHUT_BAR, "step_cap": STEP_CAP,
                          "resume_step": RESUME_STEP,
                          "D_kill_adam_A0": D_KILL_ADAM,
                          "D_kill_opt1c_annihilation": OPT1C_DKILL},
            "precedence": ("the kill wins iff its t_x precedes the "
                           "measured 2.6-crossing step (a dead read AT "
                           "the crossing wins the tie); an alive 2.6 "
                           "crossing certifies BLEED-SPARED-PAST-GATE and "
                           "the run continues to the cap — a post-gate "
                           "kill is reported as the bleed's own measured "
                           "kill-D, never re-adjudicated; a kill OUTSIDE "
                           "[2.12, 3.27] fires NO bar (graded outcome)"),
            "composite_order": "BLEED-KILLS-AT-ADAM-GATE -> "
                               "BLEED-SPARED-PAST-GATE -> CAP-AGAIN "
                               "(frozen before compute)",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1, single seed lineage (10902, A2b's; the "
                            "replicate ladder is a separate dispatch "
                            "decision); ONE root, ONE input stream (the "
                            "committed one, resume-gated bit); single "
                            "family (the e131 consolidated line); the "
                            "continuation differs from opt1b2's bleed in "
                            "NOTHING — same optimizer, same stream, same "
                            "clip — it is the same training, resumed "
                            "past its cap"),
            "extrapolation_free": ("every adjudication input is MEASURED "
                                   "inside this run's window: the kill "
                                   "(if any) is bracketed by EVERY-STEP "
                                   "reads (the bracket is ONE step wide — "
                                   "the finest this instrument has had) "
                                   "and interpolated linearly-in-step; "
                                   "the spare is certified by the read AT "
                                   "the measured 2.6-crossing step; only "
                                   "CAP-AGAIN carries the stall reading, "
                                   "GRADED by registration"),
            "float_texture": ("CPU fp32, this process, 4 threads (the "
                              "e185/opt1/opt1b/opt1b2 reduction order); "
                              "the resume gate bounds cross-process float "
                              "drift explicitly (md5 chain + bit "
                              "comparisons + the displacement recompute, "
                              "observed "
                              + ("bit" if disp_recheck_diff < G_BIT_TOL
                                 else "within tolerance") + ")"),
            "committed_data_reuse": ("opt1's Adam arms, opt1c's "
                                     "annihilation curve + densification, "
                                     "opt1b's committed first-600 curve, "
                                     "and opt1b2's committed 601..1200 "
                                     "curve are REUSED from committed "
                                     "metrics (plotted, never rerun); "
                                     "this run's reads begin at step 1201 "
                                     "and are EVERY-STEP"),
            "no_guarantees": ("nothing was guaranteed ex ante: the bleed "
                              "could die at ~2.2 (the gate reunifies), "
                              "pass 2.6 alive (separate rings), or stall "
                              "through all 150 steps (a possible diffusive "
                              "equilibrium that never dies) — the openness "
                              "is the point; the stop actually observed is "
                              f"'{stop['kind']}'"
                              + (" (post-gate)" if stop.get("post_gate")
                                 else "")),
            "recovery_provenance": ("THIRD DISPATCH (second recovery): "
                                    "dispatch-2 executed the frozen "
                                    "registration and was killed mid-flight "
                                    "at s1264 by the fifth disruption (its "
                                    "survivors: the mid-flight chunk_state "
                                    "+ the PARTIAL metrics.json committed "
                                    "at b7f8e20). This run resumed from "
                                    "that certified state "
                                    "(gates.G_MIDRESUME) and never "
                                    "re-executed steps 1201..1264; bars "
                                    "verbatim, no bar shopping. Recovery "
                                    "layer only — no bar, cadence, gate "
                                    "threshold or adjudication logic "
                                    "changed."),
            "interpolation_texture": ("the kill's t_x/D_kill are "
                                      "linear-in-step interpolations "
                                      "inside a ONE-STEP measured bracket "
                                      "(width reported); the bracket is "
                                      "the honest resolution limit — and "
                                      "it is the finest the arc has ever "
                                      "had"),
            "monitors_are_monitors": ("the diffusive-rate and "
                                      "stall-stability co-reads report "
                                      "texture; they gate nothing and "
                                      "adjudicate nothing"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": {
            fin_name: {"path": f"runs/checkpoints/{fin_name}.pt",
                       "arm": "a2b_sgd_1e-2", "steps": int(fin_step),
                       "stop": stop["kind"]},
            "chunk_state": {"path": "runs/opt1b3/chunk_state.pt",
                            "note": "the resumable continuation state "
                                    "(model + generator + step + "
                                    "continuation journal + every-step "
                                    "reads)"},
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "eval_device": "cpu", "torch_threads": 4,
                   "smoke": SMOKE, "torch": torch.__version__,
                   "chunk_cap_s": CHUNK_CAP},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    with open(rd / "journal.jsonl", "w", encoding="utf-8") as f:
        for r in journal:
            f.write(json.dumps(r) + "\n")

    # the FULL bleed curve (committed s0..s1200 + this run's every-step rows)
    curve = ([{"step": r["step"], "D": r["D"], "gm12": r["gm12"],
               "ce_r": r.get("ce_r"), "provenance": "opt1+opt1b (committed, to s600)"}
              for r in bleed_s600]
             + [{"step": r["step"], "D": r["D"], "gm12": r["gm12"],
                 "ce_r": r.get("ce_r"),
                 "provenance": "opt1b2 (committed, s601..s1200)"}
                for r in bleed_s1200]
             + [{"step": r["step"], "D": r["cum_disp"], "gm12": r["gm12"],
                 "ce_r": r["ce_r"], "provenance": "opt1b3 (this run)"}
                for r in reads])
    curve.sort(key=lambda r: r["step"])

    plot_last_steps(rd / "opt1b3_last_steps.png", reads, journal, stop,
                    verdict, clause, diffusive, stall)
    plot_four_class(rd / "opt1b3_four_class_overlay.png", curve, adam_arms,
                    anni_curve, anni_densify, reads, verdict, clause, stop)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'opt1b3_last_steps.png'}, "
        f"{rd / 'opt1b3_four_class_overlay.png'}, journal.jsonl, "
        f"chunk_state.pt, runs/checkpoints/{fin_name}.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

ADAM_STYLE = {
    "a0_adamw_ref": ("crimson", "A0 AdamW 1e-3 (t*=+2, D_kill 2.489)"),
    "a3_adamw_warmup30": ("darkorange", "A3 AdamW warmup30 (10x clock, "
                                          "D_kill 3.636)"),
    "a4_adamw_b2_0.999": ("seagreen", "A4 AdamW b2=.999 (D_kill 2.485)"),
    "a5_adamw_moment_reset": ("orchid", "A5 AdamW moment-reset (=A0)"),
}


def plot_last_steps(path, reads, journal, stop, verdict, clause,
                    diffusive, stall):
    """THE PER-STEP CURVE: g-12 every step, D(t) every step, the diffusive
    rate, the stall texture (the dispatch's named co-reads)."""
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    steps = [r["step"] for r in reads]
    gm = [r["gm12"] for r in reads]

    # (0,0) g-12 per step — the kill clock at one-step resolution
    ax = axes[0, 0]
    ax.plot(steps, gm, "o-", ms=3.2, lw=1.3, color="blue")
    ax.axhline(SHUT_BAR, ls="--", lw=1.6, color="tab:purple",
               label=f"{SHUT_BAR} DISSOLVE (the kill bar)")
    ax.axhline(SURVIVE_NOTE, ls=":", lw=1.2, color="seagreen",
               label="0.50 SPARE reference")
    ax.axhline(0.3290, ls=":", lw=1.2, color="gray", alpha=0.8,
               label="opt1b2 s1200 committed (0.3290)")
    if stop["kind"] == "kill":
        ax.axvline(stop["t_x"], ls="-", lw=2.0, color="tab:purple",
                   alpha=0.7, label=f"kill t_x {stop['t_x']:.2f}")
        ax.plot([stop["t_x"]], [SHUT_BAR], "*", ms=19, color="yellow",
                mec="k", zorder=6)
    ax.set_xlabel("wash step (continuation 1201..)")
    ax.set_ylabel("g-12 (EVERY step)")
    ax.legend(fontsize=7.2, loc="lower left")
    ax.set_title("THE PER-STEP KILL CLOCK — one-step resolution",
                 fontsize=10)

    # (0,1) g-12 vs D (the stall zone, zoomed)
    ax = axes[0, 1]
    ax.plot([r["cum_disp"] for r in reads], gm, "o-", ms=3.2, lw=1.3,
            color="blue")
    ax.axhline(SHUT_BAR, ls="--", lw=1.6, color="tab:purple",
               label=f"{SHUT_BAR} DISSOLVE")
    ax.axvspan(KILL_WINDOW[0], KILL_WINDOW[1], color="tab:purple",
               alpha=0.08, label="kill window [2.12, 3.27]")
    ax.axvline(2.12, ls=":", lw=1.4, color="purple", alpha=0.7,
               label="window lower margin (crossed ALIVE at s~1167)")
    if stop["kind"] == "kill":
        ax.plot([stop["D_kill_interp"]], [SHUT_BAR], "*", ms=19,
                color="yellow", mec="k", zorder=6,
                label=f"kill D {stop['D_kill_interp']:.4f}")
    ax.set_xlabel("cumulative displacement D")
    ax.set_ylabel("g-12")
    ax.legend(fontsize=7.2, loc="lower left")
    ax.set_title("g-12 vs D in the stall zone (inside the window, alive)",
                 fontsize=10)

    # (0,2) D(t) per step + ballistic vs diffusive references
    ax = axes[0, 2]
    ax.plot([r["step"] for r in journal], [r["cum_disp"] for r in journal],
            "-", lw=1.8, color="royalblue", label="D(t) measured")
    if journal:
        j0 = journal[0]
        t0 = j0["step"] - 1
        d0 = float(R1200_REF["cum_disp"]) if t0 == RESUME_STEP else j0["cum_disp"]
        tt = np.array([RESUME_STEP, journal[-1]["step"]])
        ref = diffusive or {}
        lin = ref.get("linear_fit", {}).get("slope_D_per_step")
        if lin:
            ax.plot(tt, d0 + lin * (tt - RESUME_STEP), "--", lw=1.2,
                    color="teal", alpha=0.85,
                    label=f"linear fit {lin:.2e}/step (r2 "
                          f"{(ref.get('linear_fit', {}).get('r2') or 0.0):.3f})")
        sq = ref.get("sqrt_fit", {}).get("slope_D_per_sqrt_step")
        if sq:
            ts = np.linspace(0, max(tt[-1] - RESUME_STEP, 1), 60)
            ax.plot(RESUME_STEP + ts, d0 + sq * np.sqrt(ts), ":",
                    lw=1.6, color="mediumvioletred", alpha=0.9,
                    label=f"SQRT fit {sq:.2e}/sqrt(step) (r2 "
                          f"{(ref.get('sqrt_fit', {}).get('r2') or 0.0):.3f})")
        msd = ref.get("mean_step_disp")
        if msd:
            ax.plot(tt, d0 + msd * (tt - RESUME_STEP), "--", lw=1.0,
                    color="darkorange", alpha=0.7,
                    label=f"ballistic ref {msd:.2e}/step (colinear)")
    ax.axhline(D_SPARE, ls=":", lw=1.6, color="navy", alpha=0.9,
               label="D=2.6 spared gate")
    ax.axhspan(ADAM_GATE_BAND[0], ADAM_GATE_BAND[1], color="tab:purple",
               alpha=0.10, label="Adam kill band 2.49-2.84")
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"$\|\theta_t-\theta_0\|_2$")
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("D(t) in the stall — diffusive vs ballistic (the co-read)",
                 fontsize=10)

    # (1,0) walked length vs projected growth (the sublinearity)
    ax = axes[1, 0]
    if len(journal) >= 2:
        dd = np.diff(np.array([r["cum_disp"] for r in journal]))
        ssteps = [r["step"] for r in journal][1:]
        ax.plot(ssteps, dd, "o-", ms=3.0, lw=1.1, color="mediumvioletred",
                label="dD per step (projected growth)")
        ax.plot([r["step"] for r in journal],
                [r["step_disp"] for r in journal], "-", lw=1.2,
                color="darkorange", alpha=0.85,
                label="step |d| (walked length)")
        if diffusive:
            ax.set_title("THE SUBLINEARITY — walked "
                         f"{diffusive['mean_step_disp']:.2e} buys "
                         f"{diffusive['mean_delta_D']:.2e} "
                         f"({diffusive['sublinearity_ratio_mean_step_over_deltaD']:.1f}x)",
                         fontsize=9.5)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("wash step")
    ax.set_ylabel("L2 per step")
    ax.legend(fontsize=7.2, loc="upper right")

    # (1,1) CE_R + alignment (every 10 steps)
    ax = axes[1, 1]
    fr = [r for r in reads if r["full"] and r["ce_r"] is not None]
    if fr:
        ax.plot([r["step"] for r in fr], [r["ce_r"] for r in fr], "s-",
                ms=5, lw=1.6, color="saddlebrown", label="CE_R (organism)")
        ax.axhline(1.6635, ls="--", lw=1.0, color="gray", alpha=0.8,
                   label="root CE_R 1.664")
        ax2 = ax.twinx()
        ax2.plot([r["step"] for r in fr],
                 [r["cos_delta_fact_g0"] for r in fr], "o--", ms=4,
                 lw=1.2, color="teal", label="cos(dtheta, grad g0)")
        ax2.plot([r["step"] for r in fr],
                 [r["cos_delta_fact_m12"] for r in fr], "x--", ms=4,
                 lw=1.0, color="olive", alpha=0.8,
                 label="cos(dtheta, grad m12)")
        ax2.axhline(0.0, color="k", lw=0.8)
        ax2.set_ylabel("alignment cos (neg = death-aligned)",
                       color="teal")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="lower left")
    ax.set_xlabel("wash step")
    ax.set_ylabel("CE_R", color="saddlebrown")
    ax.set_title("organism + alignment in the stall (every 10 steps)",
                 fontsize=10)

    # (1,2) the verdict + the stall's stability summary
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"OPT1B3 VERDICT: {verdict}", fontsize=10.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=62, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.9, va="top",
                family="monospace")
        y -= 0.024
    y -= 0.02
    if stall:
        ax.text(0.02, y, "  THE STALL (per-step g-12):", fontsize=7.4,
                va="top", family="monospace", weight="bold")
        y -= 0.026
        for ln in [
            f"   n {stall['n_step_reads']}   mean {stall['gm12_mean']:.4f}   "
            f"std {stall['gm12_std']:.4f}",
            f"   min {stall['gm12_min']:.4f}  max {stall['gm12_max']:.4f}  "
            f"margin-to-bar {stall['min_margin_to_shut_bar']:.4f}",
            f"   slope {stall['gm12_slope_per_step']:+.2e}/step   "
            f"steps<0.30: {stall['steps_below_0p30']}",
        ]:
            ax.text(0.02, y, ln, fontsize=6.9, va="top", family="monospace")
            y -= 0.024
    if diffusive:
        y -= 0.015
        ax.text(0.02, y, "  THE DIFFUSIVE RATE:", fontsize=7.4, va="top",
                family="monospace", weight="bold")
        y -= 0.026
        for ln in [
            f"   walked {diffusive['mean_step_disp']:.2e}/step buys "
            f"{diffusive['mean_delta_D']:.2e} D "
            f"({diffusive['sublinearity_ratio_mean_step_over_deltaD']:.1f}x)",
            f"   linear r2 {(diffusive['linear_fit']['r2'] or 0.0):.4f}  vs  sqrt "
            f"r2 {(diffusive['sqrt_fit']['r2'] or 0.0):.4f}",
            f"   D traveled this run {diffusive['D_traveled_this_run']:.4f}",
        ]:
            ax.text(0.02, y, ln, fontsize=6.9, va="top",
                    family="monospace")
            y -= 0.024

    fig.suptitle("OPT1B3 — THE LAST FOURTEEN STEPS: the per-step curve of "
                 "the stall (g-12 read EVERY step)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_four_class(path, curve, adam_arms, anni_curve, anni_densify,
                    my_reads, verdict, clause, stop):
    """THE FINAL FOUR-CLASS OVERLAY: guillotine (Adam arms), annihilation
    (opt1c), bleed committed (opt1+opt1b, s0-600), bleed continued
    (opt1b2 s601-1200) + this run's every-step final segment — all from
    committed data except this run's points."""
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.2),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    com600 = [c for c in curve if c["provenance"].startswith("opt1+opt1b")]
    cont = [c for c in curve if c["provenance"].startswith("opt1b2 (")]
    new = [c for c in curve if c["provenance"].startswith("opt1b3")]
    ax.plot([c["D"] for c in com600], [c["gm12"] for c in com600], "o-",
            ms=4.0, lw=1.6, color="royalblue", alpha=0.9,
            label="THE BLEED SGD 1e-2 — committed s0..600 (opt1 + opt1b)")
    ax.plot([c["D"] for c in cont], [c["gm12"] for c in cont], "s-", ms=4.0,
            lw=1.6, color="cornflowerblue", alpha=0.95,
            label="THE BLEED — committed s601..1200 (opt1b2, the walk)")
    if new:
        ax.plot([c["D"] for c in new], [c["gm12"] for c in new], "D-", ms=3.4,
                lw=1.8, color="blue", alpha=0.95, zorder=5,
                label="THE BLEED — opt1b3 final segment (this run, EVERY step)")
    for t, rows in adam_arms.items():
        col, lbl = ADAM_STYLE[t]
        pts = [(r["cum_disp"], r["gm12"]) for r in rows if r["step"] > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--", ms=5,
                lw=1.5, color=col, alpha=0.85, label=lbl)
    pts = [(r["D"], r["gm12"]) for r in anni_curve]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "X-", ms=9, lw=1.8,
            color="black", alpha=0.9, zorder=4,
            label=f"ANNIHILATION — raw-grad dir at Adam size (opt1c, "
                  f"D_kill {OPT1C_DKILL:.3f})")
    dpts = [(r["D"], r["gm12"]) for r in anni_densify]
    if dpts:
        ax.plot([p[0] for p in dpts], [p[1] for p in dpts], "x", ms=7,
                mew=2.0, color="black", alpha=0.75, zorder=4,
                label="opt1c along-path densification (f=0.2..0.8)")
    ax.axvspan(KILL_WINDOW[0], KILL_WINDOW[1], color="tab:purple",
               alpha=0.08, label="kill window [2.12, 3.27] (2.49-2.84 ±15%)")
    ax.axvline(D_SPARE, ls=":", lw=2.0, color="navy", alpha=0.8,
               label="D=2.6 spared gate")
    for yv, col, lbl in ((0.50, "seagreen", "0.50 SPARE"),
                         (SHUT_BAR, "tab:purple", f"{SHUT_BAR} DISSOLVE")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    if stop["kind"] == "kill":
        ax.plot([stop["D_kill_interp"]], [SHUT_BAR], "*", ms=19,
                color="yellow", mec="k", zorder=6,
                label=f"bleed kill (interp D {stop['D_kill_interp']:.4f}"
                      + (", POST-gate" if stop.get("post_gate")
                         else ", pre-gate") + ")")
        if stop.get("d26_row"):
            d26 = stop["d26_row"]
            ax.plot([d26["cum_disp"]], [d26["gm12"]], "*", ms=15,
                    color="lime", mec="k", alpha=0.7, zorder=6,
                    label=f"passed D=2.6 ALIVE (g-12 {d26['gm12']:.3f})")
    elif isinstance(stop, dict) and stop.get("d26_row"):
        d26 = stop["d26_row"]
        ax.plot([d26["cum_disp"]], [d26["gm12"]], "*", ms=19,
                color="lime", mec="k", zorder=6,
                label=f"passed D=2.6 ALIVE (g-12 {d26['gm12']:.3f})")
    ax.set_xlabel(r"cumulative displacement $\|\theta_t-\theta_0\|_2$ "
                  "(2.74M params)")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.08, 5.3)
    ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("THE FINAL FOUR-CLASS OVERLAY — the bleed's final segment "
                 "added to the committed classes (guillotine / "
                 "annihilation / bleed-committed / bleed-continued)",
                 fontsize=9.0)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"OPT1B3 VERDICT: {verdict}", fontsize=10.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.05
    for wd in textwrap.wrap(clause, width=86, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=7.2, va="top",
                family="monospace")
        y -= 0.026
    y -= 0.02
    ax.text(0.02, y, "  THE DOOR (opt1b2, committed): kill extrapolated "
                     f"s{THE_DOOR['extrapolated_kill_step']:.1f}",
            fontsize=7.2, va="top", family="monospace", color="gray")
    y -= 0.03
    ax.text(0.02, y, "  step     D        g-12    g0     CE_R    cos(g0)  "
                     " forced", fontsize=7.2, va="top", family="monospace")
    y -= 0.028
    fr = [r for r in my_reads if r["full"]]
    tail = fr[-26:]
    for r in tail:
        cos = r["cos_delta_fact_g0"]
        ax.text(0.02, y,
                f"  {r['step']:>5d}  {r['cum_disp']:7.4f}  {r['gm12']:.4f}"
                f"  {(r['g0'] if r['g0'] is not None else float('nan')):.4f}"
                f"  {(r['ce_r'] if r['ce_r'] is not None else float('nan')):.4f}"
                f"  " + (f"{cos:+.4f}" if cos is not None else "    n/a")
                + f"  {r['forced'] or ''}",
                fontsize=6.9, va="top", family="monospace",
                color=("blue" if r["forced"] else "black"))
        y -= 0.026
    if len(fr) > len(tail):
        ax.text(0.02, y, f"  … ({len(fr) - len(tail)} earlier full reads "
                         "in metrics.json; g-12 at EVERY step in "
                         "fact_vs_D_curve_this_run)",
                fontsize=6.6, va="top", family="monospace", color="gray")
    fig.suptitle("OPT1B3 — THE FINAL FOUR-CLASS OVERLAY: the trajectory "
                 "classes at their own gates, completed", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
