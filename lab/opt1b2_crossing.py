"""OPT1B2 — THE GATE-CROSSING READ (T142/T143's cell; the bleed's own gate).

WHY: the bleed (SGD lr 1e-2 on the licensed e185 wash cell — the
raw-gradient direction at its own small step size) was stopped ALIVE
(g-12 0.601) at D 1.43 by opt1b's frozen 600-step cap; its labeled
projection reads D=2.6 at ~step 761 ALIVE and the kill at ~step 1829
(D ~ 10). opt1c then showed the raw-gradient DIRECTION at Adam's step
size kills at D 0.920 — 2.7x BELOW the Adam gate — while the bleed
passed 0.92 ALIVE holding ~0.79: re-orientation spares it. THE OPEN
QUESTION this cell owns: where does the re-orienting path ACTUALLY
die? The Adam gate is 2.49-2.84; the annihilation is 0.920; the
projection says ~10. The direct read decides.

THE CELL: CONTINUE the bleed from its committed s600 state (the
committed runs/opt1b/chunk_state.pt — model + generator state + step
+ journal, the state a cross-process resumer would load — cross-gated
bit vs runs/checkpoints/opt1b_a2b_sgd_1e-2_s600.pt) with opt1b's
machinery VERBATIM (plain SGD m=0/wd=0/clip 1.0, constant lr 1e-2,
the seed-10902 licensed stream whose position carries in the saved
generator state), to the kill (g-12 <= 0.27) or D = 3.3 reached alive
or a 1200-step cap — whichever FIRST. Reads at FINE cadence AROUND
the D=2.6 crossing (every 10 steps while 2.2 <= D <= 3.0); the
standard adaptive cadence elsewhere (opt1b's 40/10/4 by g-12
0.55/0.35); FORCED reads at the measured 2.6- and 3.3-crossing steps.

REGISTERED BARS (frozen here before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - BLEED-KILLS-AT-ADAM-GATE: "fires if the bleed dies within
    [2.12, 3.27] — the re-orienting path reaches the same gate as the
    guillotine; 'the gate is raw displacement' survives cross-class
    (reunifying e188 within AND across classes at the gate
    magnitude)."
  - BLEED-SPARED-PAST-GATE: "fires if the bleed passes D = 2.6 alive
    (g-12 > 0.27) — the gate is trajectory-class-typed all the way;
    the law's final form names the classes (guillotine ~2.5 /
    annihilation ~0.9 / bleed its own, reported with the measured
    kill-D)."
  - CAP-NEITHER: "neither within 1200 steps — the curve + labeled
    projection reported; no adjudication."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses,
they do not move the bars):
  * the fact = g-12 (install-60 battery mean p(Z) at ctx offset -12,
    e185's convention); KILL = g-12 <= 0.27 (the arc's SHUT bar),
    interpolated linearly-in-step between the last alive read and the
    first dead read (opt1b's convention); D_kill = the same-bracket
    linear interpolation of cumulative displacement; the fine cadence
    bounds the kill bracket at 10 steps inside the crossing zone.
  * displacement = cumulative ||theta_t - theta_0||_2 over all
    2,739,072 params from the ROOT (fp32, CPU, measured every step) —
    NOT from the s600 resume point; e185/opt1/opt1b's currency
    verbatim, so D is comparable across all four classes.
  * D = 2.6 = the SPARED gate (opt1b/opt1c's frozen number), reached
    = the first step whose cumulative displacement crosses it (a
    FORCED read fires there); D = 3.3 = the alive-stop (the kill
    window's top 3.27 + margin; opt1c's number).
  * precedence (opt1b's frozen convention, extended to the 3.3 stop):
    the kill wins iff its interpolated t_x precedes the measured
    2.6-crossing step (a dead read AT the crossing step wins the tie
    — opt1c's "kill first at a tie step"); an ALIVE forced read at
    the 2.6 crossing CERTIFIES BLEED-SPARED-PAST-GATE, and the run
    CONTINUES to 3.3 / a post-gate kill / the cap per the stop rule —
    a kill AFTER a certified live crossing does NOT un-fire SPARED
    (whichever-first in time); its measured D_kill is REPORTED as the
    bleed's own gate (the SPARED bar's own letter) without
    re-adjudication. A kill before the crossing adjudicates the KILLS
    bar against [2.12, 3.27]; a kill OUTSIDE the window fires NO bar
    (opt1b's convention: graded outcome, curve reported verbatim).
  * cap = 1200 steps TOTAL from the root (the dispatch's 1200), i.e.
    600 continuation steps; CAP-NEITHER's projections are labeled
    EXTRAPOLATED and never adjudicate.
  * the arm guarantees NOTHING (stated before compute): it could die
    at 2.5, pass the gate alive and die at 3.0, reach 3.3 alive, or
    creep to the cap — the openness is the point.

RESUME / STREAM-CONTINUITY METHOD (the dispatch's letter): the bleed's
committed s600 state is runs/opt1b/chunk_state.pt (saved by opt1b at
its cap stop: model + generator state + step 600 + the 600-row journal
+ the 14 reads) — the exact state a cross-process resumer loads; the
final checkpoint runs/checkpoints/opt1b_a2b_sgd_1e-2_s600.pt holds the
same model (gated BIT-IDENTICAL here). Plain SGD(momentum 0) is
STATELESS — no optimizer state to carry; the stream position carries
in the GENERATOR STATE, so the input stream CONTINUES bit-identically
from step 601, never repeats (per-step md5s recorded; battery reads
consume no RNG). The RESUME GATE (Rule 12): the loaded model's flat
md5 == opt1b's committed hash-chain tail, the s600.pt model == the
chunk_state model (bit), the loaded journal == the committed
runs/opt1b/journal.jsonl (row-equal) with its last row hard-bound to
the committed step-600 trajectory row, the recomputed cumulative
displacement ||theta_600 - theta_root|| == the committed journal's
step-600 value (bit tolerance), the committed journal's steps 1..10
x_md5s == e185's stored hashes, and every NEW step's x_md5 unique vs
the whole committed+new stream — the first resumed step continues the
committed trajectory bit-consistently where comparable.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any continuation): the
standard cell gates (corpus ZEPH count 0; splice mix {FLORIZEL: 19,
ELIZABETH: 41}; battery shapes 60 x (130 +- j); neutral bank 16 starts
bit-equal to e185's stored list; root gated vs e151's before-cells)
PLUS the resume gate above.

CO-READS: (1) D(t) and g-12-vs-D overlaid on ALL classes from
COMMITTED data (opt1's four Adam arms, opt1c's annihilation arm with
its along-path densification, opt1b's committed first-600 bleed curve
— loaded and plotted, never rerun) + this continuation; (2) CE_R at
every checkpoint (organism health); (3) alignment cos(delta_theta_t,
grad g0_t) and cos(delta_theta_t, grad m12_t) at every checkpoint
(critic's sign: negative = death-aligned); (4) THE PUMP-CLIFF MONITOR
— the per-interval slope d(g-12)/dD ladder along the FULL bleed
(committed + continuation): WHERE does the fact's decay accelerate?
(the pump-cliff signature opt1c saw as pump-to-0.955 at D~0.33 then
cliff-by-D~1.0 on the big-step path; along the bleed the same
terrain, taken slowly). Monitor only — it never adjudicates.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is g1bS's — never claimed), torch threads 4
(the e185/opt1 reduction order), ckpt-RESUMABLE chunks <= 180 s per
the opt1b convention (full state round-trips through runs/opt1b2/
chunk_state.pt after each chunk; each chunk LOADS the previous chunk's
file; chunk-boundary parameter md5s form a hash chain), PROGRESSIVE
PARTIAL metrics.json writes (the 2026-09-30 outage lesson, opt1c's
recovery convention: after the gates pass and after every chunk
save), a completed-run RECOVERY HOOK (stop persisted in chunk_state;
never steps past a completed kill), n=1, single seed lineage (10902),
no reruns beyond the cap.

Outputs: runs/opt1b2/{metrics.json, opt1b2_fact_vs_D.png,
opt1b2_coreads.png, chunk_state.pt, journal.jsonl (continuation rows
601.. only; the committed 600-row parent journal is referenced by
md5)}; checkpoint runs/checkpoints/opt1b2_a2b_sgd_1e-2_s<final>.pt.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python opt1b2_crossing.py    (OPT1B2_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185/e187/opt1/opt1b/opt1c)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185/opt1/opt1b-era reduction order

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("OPT1B2_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "opt1b2 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (opt1/opt1b's, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E185_METRICS = E43.REPO / "runs" / "e185" / "metrics.json"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
OPT1B_METRICS = E43.REPO / "runs" / "opt1b" / "metrics.json"
OPT1B_CHUNK = E43.REPO / "runs" / "opt1b" / "chunk_state.pt"
OPT1B_JOURNAL = E43.REPO / "runs" / "opt1b" / "journal.jsonl"
OPT1B_S600 = CKPT_DIR / "opt1b_a2b_sgd_1e-2_s600.pt"
OPT1C_METRICS = E43.REPO / "runs" / "opt1c" / "metrics.json"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the arm + the continuation envelope (dispatch-frozen) -----------------------
LR_A2B = 1e-2                     # opt1 A2b / opt1b's lr VERBATIM
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185/opt1/opt1b)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
CHUNK_CAP = 178.0                 # dispatch: <=180 s per chunk (margin for step overshoot)
STEP_CAP = 1200                   # dispatch: 1200-step cap TOTAL from the root
D_SPARE = 2.6                     # the SPARED gate (opt1b/opt1c's frozen number)
D_STOP = 3.3                      # the alive-stop (kill window top 3.27 + margin; opt1c's)
KILL_WINDOW = (2.12, 3.27)        # 2.49-2.84 +- 15% (the registered letter's window)
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE)
SURVIVE_NOTE = 0.50               # SPARE reference line (opt1's convention; report-only)
E170_ANCHOR_SEED = 170            # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
ADAM_GATE_BAND = (2.49, 2.84)     # opt1's measured Adam kill displacements (T139)
OPT1C_DKILL = 0.9203406595225093  # opt1c's annihilation kill displacement (overlay reference)

# ---- read cadence (dispatch): FINE around the crossing, standard elsewhere -------
FINE_ZONE = (2.2, 3.0)            # every FINE_GAP steps while 2.2 <= D <= 3.0
FINE_GAP = 10
GAP_HI, GAP_MID, GAP_LO = 40, 10, 4   # opt1b's standard adaptive gaps (0.55/0.35 triggers)
if SMOKE:                         # true shakedown trims (documented in deviations)
    STEP_CAP, CHUNK_CAP, FINE_GAP = 604, 12.0, 2
    GAP_HI = GAP_MID = GAP_LO = 2

# ---- the committed s600 resume references (hard-bound at runtime) ----------------
RESUME_STEP = 600
RESUME_MD5_600 = "39f0050d23e33c5a12603a21882b3041"   # opt1b's committed hash-chain tail
J600_REF = {                      # opt1b's committed journal row at step 600 (full precision)
    "step": 600, "chunk": 7,
    "ce_batch": 0.8501756194563662,
    "cum_disp": 1.4338268041610718,
    "step_disp": 0.006714710994430027,
    "preclip_gnorm": 0.671520571572876,
    "lr_eff": 0.01,
    "x_md5": "5cef62eed376621635e92086d0ef590c",
}
R600_REF = {                      # opt1b's committed read at step 600 (the read anchor)
    "step": 600, "gm12": 0.6012769937515259, "g0": 0.5320451259613037,
    "ce_r": 1.7218157052993774, "cum_disp": 1.4338268041610718,
}

# ---- gates / references (full precision, = stored metrics; opt1's set) ----------
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
    "row0_strength": 0.7316772222270098,
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
    "bleed_kills_at_adam_gate": "BLEED-KILLS-AT-ADAM-GATE: \"fires if the "
        "bleed dies within [2.12, 3.27] — the re-orienting path reaches the "
        "same gate as the guillotine; 'the gate is raw displacement' "
        "survives cross-class (reunifying e188 within AND across classes "
        "at the gate magnitude).\"",
    "bleed_spared_past_gate": "BLEED-SPARED-PAST-GATE: \"fires if the bleed "
        "passes D = 2.6 alive (g-12 > 0.27) — the gate is trajectory-class-"
        "typed all the way; the law's final form names the classes "
        "(guillotine ~2.5 / annihilation ~0.9 / bleed its own, reported "
        "with the measured kill-D).\"",
    "cap_neither": "CAP-NEITHER: \"neither within 1200 steps — the curve + "
        "labeled projection reported; no adjudication.\"",
    "operationalizations": "the arm = opt1b's committed s600 bleed state "
        "CONTINUED verbatim (plain SGD m0/wd0/clip1.0, constant lr 1e-2, "
        "the seed-10902 licensed stream whose position carries in the saved "
        "generator state; resume gated bit vs the committed chunk_state + "
        "s600.pt + journal); KILL = g-12 <= 0.27 interpolated linearly-"
        "in-step between the last alive and first dead read (the fine "
        "cadence — every 10 steps while 2.2 <= D <= 3.0 — bounds the "
        "bracket); D_kill = same-bracket linear interpolation of cumulative "
        "displacement FROM THE ROOT; D = 2.6 'passed alive' = a FORCED read "
        "at the first step whose cumulative displacement crosses 2.6 shows "
        "g-12 > 0.27; stop = whichever FIRST of kill / D >= 3.3 alive "
        "(forced read) / 1200-step cap; precedence (opt1b's frozen "
        "convention): the kill wins iff its t_x precedes the measured "
        "2.6-crossing step (a dead read AT the crossing wins the tie); an "
        "alive 2.6 crossing CERTIFIES SPARED and the run CONTINUES per the "
        "stop rule — a post-gate kill is REPORTED as the bleed's own "
        "measured kill-D (the SPARED bar's own letter), never "
        "re-adjudicated; a kill OUTSIDE [2.12, 3.27] fires NO bar (graded "
        "outcome, curve reported verbatim); composite order frozen "
        "BLEED-KILLS-AT-ADAM-GATE -> BLEED-SPARED-PAST-GATE -> CAP-NEITHER.",
    "registration": "the dispatch's registration IS the registration (the "
        "three bars quoted verbatim in the module docstring and here, "
        "frozen before compute). Adjudicate against exactly this; no bar "
        "shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "PLOT/KEY FIX (caught by this run's own smoke): opt1b's committed "
    "fact_vs_D_curve rows key displacement as 'D' while this run's "
    "ckpt rows use 'cum_disp' — the same key-fix class as opt1c's "
    "recovery; the curve builder reads the committed key. Data layer "
    "only — no arm/gate/bar logic touched.",
    "THE RESUME: the run does NOT re-execute steps 1..600 — it loads "
    "opt1b's committed chunk_state (the state a cross-process resumer "
    "would take), gated bit (flat md5 == the committed hash-chain tail; "
    "s600.pt model bit-identical; journal row-equal to runs/opt1b/"
    "journal.jsonl with the step-600 row hard-bound; recomputed "
    "displacement bit-tolerant vs the committed value). The continuation "
    "journal written to runs/opt1b2/journal.jsonl contains ONLY the new "
    "rows (601..); the parent's 600 rows are referenced by their file "
    "md5 + the embedded hard-bound row.",
    "Read cadence (the dispatch's letter): every 10 steps while "
    "2.2 <= cum_disp <= 3.0 (the FINE zone around the projected 2.6 "
    "crossing); opt1b's standard adaptive cadence elsewhere (40 -> 10 -> 4 "
    "as g-12 falls through 0.55/0.35); FORCED reads at the measured 2.6- "
    "and 3.3-crossing steps; a final read at the cap. The first "
    "continuation read falls one standard gap (40) after the committed "
    "step-600 read (g-12 0.601 -> GAP_HI).",
    "Chunk cap 178 s (dispatch <=180 s; the margin absorbs one ~2.5 s step "
    "overshoot so every chunk's wall stays <= 180 s); in-chunk checkpoint "
    "reads count toward the cap (opt1's FT_TIME_CAP convention). Chunk "
    "state round-trips through runs/opt1b2/chunk_state.pt per the opt1b "
    "convention (model + generator state + step + continuation journal + "
    "reads + flags); each chunk LOADS the previous chunk's file; "
    "chunk-boundary parameter md5s form a hash chain.",
    "PROGRESSIVE PARTIAL metrics.json WRITES (the 2026-09-30 outage "
    "lesson; opt1c's recovery convention, mandated by this dispatch): a "
    "partial is written (a) immediately after the gates pass and (b) "
    "after every chunk save, each stamped status/phase/timestamp with "
    "gates + provenance + journal/read tails + chunks + stop-so-far; the "
    "final COMPLETE write replaces it.",
    "RESUME HARDENING (opt1c's recovery convention): chunk_state also "
    "persists stop + chunks_prov (appended BEFORE the save); a RECOVERY "
    "HOOK before the arm loop restores an already-completed run (stop set "
    "in chunk_state) so final metrics/plots regenerate WITHOUT stepping "
    "past a completed kill.",
    "Light reads only (g-12/g0 batteries + CE_R + alignment + "
    "displacement + pre-clip grad norms + the pump-cliff slope monitor) — "
    "the dispatch's read list; no census/deletions. CPU-ONLY "
    "(CUDA_VISIBLE_DEVICES=-1 before torch; the GPU is g1bS's, never "
    "claimed); threads 4; n=1; single seed lineage (10902); no reruns "
    "beyond the cap.",
    "Smoke mode trims: cap 604 (4 continuation steps), 12 s chunks, all "
    "gaps 2; the FULL parent resume gate still runs (that is what the "
    "shakedown exists to prove). Nothing adjudicated (verdict stamped "
    "SMOKE).",
    "Chunk walls: the chunk budget (178 s) is checked BETWEEN steps; "
    "under heavy outside CPU contention a single in-flight step can "
    "overshoot, so individual chunk walls may slightly exceed 180 s "
    "(observed max recorded in gates.G_CHUNKS). This is wall-clock "
    "contention, not compute: one training step is ~2.5 s of CPU on 4 "
    "threads (opt1b's own rate).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/opt1b_sgd_kill.py VERBATIM (whose own provenance is
# lab/opt1_optimizer_controls.py via lab/e185_noise_wash.py — the e176n
# lineage; opt1c's recovery-hardened chunk loop conventions adopted).
# Copied rather than imported to own the device policy.

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


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("opt1b2_smoke" if SMOKE else "opt1b2")
    chunk_path = rd / "chunk_state.pt"
    log(f"OPT1B2 THE GATE-CROSSING READ (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), sequential "
        f"chunks <= {CHUNK_CAP + 2:.0f}s, cap {STEP_CAP} steps total from "
        f"the root, n=1, seed lineage {FREEZE_SEED} (A2b's)")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (OPT1_METRICS, OPT1B_METRICS, OPT1C_METRICS, OPT1B_CHUNK,
              OPT1B_JOURNAL, OPT1B_S600):
        if not p.exists():
            raise RuntimeError(f"missing parent artifact: {p}")
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    opt1bm = json.loads(OPT1B_METRICS.read_text(encoding="utf-8"))
    opt1cm = json.loads(OPT1C_METRICS.read_text(encoding="utf-8"))
    adam_arms = {t: opt1m["arms"][t]["ckpt_table"]
                 for t in ("a0_adamw_ref", "a3_adamw_warmup30",
                           "a4_adamw_b2_0.999", "a5_adamw_moment_reset")}
    bleed_committed = opt1bm["fact_vs_D_curve"]      # the bleed, committed (to s600)
    anni_curve = opt1cm["fact_vs_D_curve"]           # opt1c's annihilation arm
    anni_densify = opt1cm["arm"].get("densify", [])  # its along-path profile
    chain_tail = opt1bm["resume"]["state_hash_chain"][-1]
    parent_journal_md5 = hashlib.md5(OPT1B_JOURNAL.read_bytes()).hexdigest()
    # hard-bind this file's embedded resume references to the COMMITTED data
    assert chain_tail == RESUME_MD5_600, \
        "opt1b's committed hash-chain tail drifted vs this file's copy"
    b2b_parent_verdict = opt1bm["adjudication"]["verdict"]
    log(f"parents: opt1 {opt1m['adjudication']['verdict']}; opt1b "
        f"{b2b_parent_verdict} (committed bleed curve "
        f"{len(bleed_committed)} rows; journal md5 {parent_journal_md5[:10]}…); "
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
    # THE RESUME GATE (Rule 12) — load opt1b's committed s600 chunk state
    # and certify it bit: chunk_state + s600.pt + committed journal.
    # =====================================================================
    pst = torch.load(OPT1B_CHUNK, map_location="cpu", weights_only=False)
    ck600 = torch.load(OPT1B_S600, map_location="cpu", weights_only=False)
    j600 = dict(pst["journal"][-1])
    r600 = dict(pst["reads"][-1])
    j600_file = [json.loads(l) for l in
                 OPT1B_JOURNAL.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    s600_bit = all(torch.equal(pst["model"][k], ck600["model"][k])
                   for k in pst["model"])
    j600_diffs = {k: abs(float(j600[k]) - float(J600_REF[k]))
                  for k in ("ce_batch", "cum_disp", "step_disp",
                            "preclip_gnorm")}
    r600_diffs = {k: abs(float(r600[k]) - float(R600_REF[k]))
                  for k in ("gm12", "g0", "ce_r", "cum_disp")}
    journal_file_equal = bool(j600_file == list(pst["journal"]))
    committed_x_md5_1_10 = {int(r["step"]): r["x_md5"] for r in j600_file}
    xmd5_1_10_ok = all(committed_x_md5_1_10.get(s) == h
                       for s, h in E185_XHASH.items())
    G_RESUME = {
        "loaded_from": "runs/opt1b/chunk_state.pt (opt1b's committed cap "
                       "stop: model + generator state + step 600 + 600-row "
                       "journal + 14 reads)",
        "chunk_step": int(pst["step"]),
        "chunk_meta": E43.jsonable(pst["meta"]),
        "flat_md5_loaded_vs_committed_chain_tail": bool(
            chain_tail == RESUME_MD5_600 == pst["flat_md5"]),
        "s600_pt_model_bit_identical": bool(s600_bit),
        "journal_file_row_equal": journal_file_equal,
        "journal_file_md5": parent_journal_md5,
        "step600_row_hard_bind_diffs": j600_diffs,
        "read600_row_hard_bind_diffs": r600_diffs,
        "committed_journal_x_md5_1_10_vs_e185": bool(xmd5_1_10_ok),
        "max_abs_diff": max(max(j600_diffs.values()),
                            max(r600_diffs.values())),
        "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
        "bit": bool(max(max(j600_diffs.values()),
                        max(r600_diffs.values())) < G_BIT_TOL),
        "pass": bool(int(pst["step"]) == RESUME_STEP
                     and chain_tail == RESUME_MD5_600 == pst["flat_md5"]
                     and s600_bit and journal_file_equal
                     and xmd5_1_10_ok
                     and max(max(j600_diffs.values()),
                             max(r600_diffs.values())) < G_FALLBACK_TOL),
        "note": "PRE-DISPATCH CHECK (Rule 12): the resumed state's identity "
                "— the committed chunk_state (what a cross-process resumer "
                "loads) vs opt1b's committed s600.pt model (bit), the "
                "committed journal.jsonl (row-equal) with the step-600 "
                "trajectory row and read hard-bound to this file's embedded "
                "full-precision copies, and the committed stream's steps "
                "1..10 x_md5s vs e185's stored hashes. The first resumed "
                "step (601) continues the committed trajectory "
                "bit-consistently: the state is the committed one (md5 "
                "chain) and the stream position carries in the generator "
                "state (per-step md5s recorded; never repeats — "
                "gates.G_INPUTS).",
    }
    log(f"G_RESUME (opt1b s600 chunk_state + s600.pt + journal): step "
        f"{pst['step']}, md5-chain "
        f"{'OK' if G_RESUME['flat_md5_loaded_vs_committed_chain_tail'] else 'MISMATCH'}, "
        f"s600.pt {'BIT' if s600_bit else 'DIFFERS'}, journal file "
        f"{'row-equal' if journal_file_equal else 'DIFFERS'}, max|d| "
        f"{G_RESUME['max_abs_diff']:.2e}: "
        + ("PASS" if G_RESUME["pass"] else "FAIL")
        + (" (bit)" if G_RESUME["bit"] else ""))
    if not G_RESUME["pass"]:
        raise RuntimeError("resume gate FAILED — the s600 state is not "
                           "certified; abort (control failure)")

    # install the resumed state
    net = copy.deepcopy(net0)
    net.load_state_dict(pst["model"])
    net.train()
    gen = torch.Generator()
    gen.set_state(pst["gen_state"])
    evl = copy.deepcopy(net0)
    evl.eval()
    prev = flat_params(net)          # displacement continuity anchor @ s600
    # the displacement recompute (same reduction order, this process)
    disp_recheck = float(torch.norm(prev - theta0))
    disp_recheck_diff = abs(disp_recheck - float(j600["cum_disp"]))
    G_RESUME["disp_recompute_measured"] = disp_recheck
    G_RESUME["disp_recompute_vs_committed_diff"] = disp_recheck_diff
    G_RESUME["disp_recompute_bit"] = bool(disp_recheck_diff < G_BIT_TOL)
    if disp_recheck_diff >= G_FALLBACK_TOL:
        raise RuntimeError("displacement recompute at resume FAILED")
    log(f"G_RESUME disp recompute: {disp_recheck:.16f} vs committed "
        f"{j600['cum_disp']:.16f} (|d| {disp_recheck_diff:.2e}, "
        f"{'bit' if disp_recheck_diff < G_BIT_TOL else 'tol'})")

    opt = torch.optim.SGD(net.parameters(), lr=LR_A2B)   # plain: m=0, wd=0
    step = int(pst["step"])
    log("WHAT THIS ARM GUARANTEES: NOTHING — it could die at 2.5, pass the "
        "gate alive and die at 3.0, reach 3.3 alive, or creep to the cap; "
        "the openness is the point.")

    # ---- PROGRESSIVE PARTIAL WRITE #1: gates passed, continuation starting
    def write_partial(status: str, stop_: dict | None) -> None:
        save_json(rd / "metrics.json", E43.jsonable({
            "experiment": "opt1b2_crossing", "date": common.now_iso(),
            "status": status, "partial": True,
            "phase": {"step": int(step), "chunks": len(chunks_prov),
                      "reads": len(reads), "journal_rows": len(journal),
                      "stop_kind": (stop_ or {}).get("kind"),
                      "elapsed_s": round(time.time() - T0, 1)},
            "gates_partial": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                              "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                              "G_ROOT": G_ROOT, "G_RESUME": G_RESUME},
            "provenance_partial": {
                "resume": ("continued from runs/opt1b/chunk_state.pt "
                           f"@ step {RESUME_STEP} (gated bit: md5 chain + "
                           "s600.pt + journal hard-bind); parent journal "
                           f"md5 {parent_journal_md5}"),
                "parents": {"opt1": b2b_parent_verdict,
                            "opt1c": opt1cm["adjudication"]["verdict"]}},
            "chunks": chunks_prov,
            "journal_tail": journal[-8:], "reads_tail": reads[-8:],
            "stop_partial": stop_,
        }))

    # =====================================================================
    # THE CHUNKED CONTINUATION (601.. to the stop)
    # =====================================================================
    journal: list[dict] = []          # CONTINUATION rows only (601..)
    reads: list[dict] = []            # continuation checkpoint reads
    chunks_prov: list[dict] = []
    zeph_checks = 0
    stop: dict | None = None
    d26_crossed = False
    d33_crossed = False
    d26_row: dict | None = None
    G_CHUNKS = {"round_trips": [], "pass": None}
    chunk_idx = 0
    # the read anchor: the committed step-600 read (for the first gap only)
    last_read_step = RESUME_STEP
    last_read_gm12 = float(r600["gm12"])

    # ---- RECOVERY HOOK (opt1c's convention): a previous process COMPLETED
    # the arm (stop set in this run's chunk_state) and died before the
    # final write -> restore, skip training, regenerate outputs.
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
            d33_crossed = bool(_probe["d33_crossed"])
            d26_row = _probe["d26_row"]
            zeph_checks = int(_probe["zeph_checks"])
            chunks_prov = list(_probe.get("chunks_prov", []))
            stop = dict(_probe["stop"])
            chunk_idx = int(_probe["chunk"]) + 1
            prev = flat_params(net)
            last_read_step = (reads[-1]["step"] if reads else RESUME_STEP)
            last_read_gm12 = (reads[-1]["gm12"] if reads
                              else float(r600["gm12"]))
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

    def std_gap() -> int:
        """opt1b's standard adaptive cadence (from the last read's g-12)."""
        if last_read_gm12 is None or last_read_gm12 >= 0.55:
            return GAP_HI
        if last_read_gm12 >= 0.35:
            return GAP_MID
        return GAP_LO

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
            d33_crossed = bool(st["d33_crossed"])
            d26_row = st["d26_row"]
            # assign, never accumulate (opt1c's recovery fix)
            zeph_checks = int(st["zeph_checks"])
            chunks_prov = list(st.get("chunks_prov", []))
            prev = flat_params(net)          # displacement continuity anchor
            last_read_step = (reads[-1]["step"] if reads else RESUME_STEP)
            last_read_gm12 = (reads[-1]["gm12"] if reads
                              else float(r600["gm12"]))
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
        else:
            log(f"[chunk {chunk_idx}] fresh start from the parent's "
                f"committed s600 state (step {step})")

        t_chunk = time.time()

        # ---- one chunk: steps until stop / cap / chunk-budget
        while stop is None:
            if step >= STEP_CAP:
                stop = {"kind": "cap", "step": step,
                        "note": "1200-step cap reached with neither bar's "
                                "trigger inside the window"}
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

            # ---- checkpoint read (fine zone / standard / forced / at-cap)
            in_fine = FINE_ZONE[0] <= cum_disp <= FINE_ZONE[1]
            gap = FINE_GAP if in_fine else std_gap()
            due = (step - last_read_step) >= gap
            forced_flags = []
            if (not d26_crossed) and cum_disp >= D_SPARE:
                d26_crossed = True
                forced_flags.append("d26")
            if (not d33_crossed) and cum_disp >= D_STOP:
                d33_crossed = True
                forced_flags.append("d33")
            forced = "+".join(forced_flags) if forced_flags else None
            at_cap = step >= STEP_CAP
            if due or forced or (at_cap and step != last_read_step):
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                delta = cur - theta0
                gz = battery_cell(evl, gm12_ids, zid)
                gz0 = battery_cell(evl, g0_ids, zid)
                ce_r = ce_fixed_cpu(evl, *r_eval_xy)
                g_g0 = fact_grad(evl, g0_ids, zid)
                g_m12 = fact_grad(evl, gm12_ids, zid)
                cos_g0 = float(torch.dot(delta, g_g0)
                               / (torch.norm(delta) * torch.norm(g_g0) + 1e-30))
                cos_m12 = float(torch.dot(delta, g_m12)
                                / (torch.norm(delta) * torch.norm(g_m12) + 1e-30))
                rrow = {"step": step, "chunk": chunk_idx, "forced": forced,
                        "gm12": gz["mean_pz"], "g0": gz0["mean_pz"],
                        "frac_argmax_z": gz["frac_argmax_z"],
                        "ce_r": ce_r, "cum_disp": cum_disp,
                        "cos_delta_fact_g0": cos_g0,
                        "cos_delta_fact_m12": cos_m12}
                reads.append(rrow)
                last_read_step = step
                last_read_gm12 = gz["mean_pz"]
                if "d26" in forced_flags:
                    d26_row = dict(rrow)
                log(f"  CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                    f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| "
                    f"{cum_disp:.4f} cos(g0) {cos_g0:+.3f}"
                    + (f"  <= D={D_SPARE} FORCED" if "d26" in forced_flags
                       else "")
                    + (f"  <= D={D_STOP} FORCED" if "d33" in forced_flags
                       else ""))
                # ---- stop checks (kill first at a tie step; composite order)
                if gz["mean_pz"] <= SHUT_BAR:
                    prevr = reads[-2] if len(reads) >= 2 else {
                        "step": RESUME_STEP, "gm12": float(r600["gm12"]),
                        "cum_disp": float(r600["cum_disp"])}
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
                            "d26_row": (dict(d26_row) if d26_row else None),
                            "post_gate": bool(d26_row is not None
                                              and d26_row["gm12"] > SHUT_BAR)}
                    break
                if d33_crossed and gz["mean_pz"] > SHUT_BAR:
                    stop = {"kind": "spared", "step": step,
                            "gm12_at_d33": gz["mean_pz"],
                            "g0_at_d33": gz0["mean_pz"],
                            "ce_r_at_d33": ce_r, "D_at_d33": cum_disp,
                            "d26_row": (dict(d26_row) if d26_row else None)}
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
                   "d33_crossed": bool(d33_crossed),
                   "d26_row": d26_row,
                   "zeph_checks": int(zeph_checks),
                   "stop": stop,
                   "chunks_prov": list(chunks_prov),
                   "flat_md5": fp_md5,
                   "meta": {"experiment": "opt1b2", "arm": "a2b_sgd_1e-2",
                            "lr": LR_A2B, "input_seed": FREEZE_SEED,
                            "base": f"runs/checkpoints/{ROOT_CK}",
                            "parent": "runs/opt1b/chunk_state.pt @ s600 "
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
                    "note": "1200-step cap reached with neither bar's "
                            "trigger inside the window"}
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
    committed_md5s = [r["x_md5"] for r in j600_file]
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
                "hashes + the row-equal journal); every NEW step's x_md5 "
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
        f"opt1b2_a2b_sgd_1e-2_s{fin_step}"
    torch.save({"model": {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()},
                "meta": {"experiment": "opt1b2", "arm": "a2b_sgd_1e-2",
                         "steps": int(fin_step), "lr": LR_A2B,
                         "input_seed": FREEZE_SEED,
                         "stop": stop["kind"],
                         "base": f"runs/checkpoints/{ROOT_CK}",
                         "parent": "runs/opt1b/chunk_state.pt @ s600 "
                                   "+ runs/opt1b2/chunk_state.pt"}},
               CKPT_DIR / f"{fin_name}.pt")
    log(f"[ckpt] saved {fin_name}.pt (stop={stop['kind']} @ s{fin_step})")

    # =====================================================================
    # THE FULL BLEED CURVE (committed 0..600 + this continuation) + the
    # pump-cliff monitor along it
    # =====================================================================
    # NOTE the key: opt1b's committed fact_vs_D_curve rows key displacement
    # as "D" (opt1c's recovery fix class — this run's ckpt rows use
    # "cum_disp"); read the committed key.
    curve = [{"step": r["step"], "D": r["D"], "gm12": r["gm12"],
              "g0": r["g0"], "ce_r": r["ce_r"],
              "cos_delta_fact_g0": r.get("cos_delta_fact_g0"),
              "cos_delta_fact_m12": r.get("cos_delta_fact_m12"),
              "provenance": "opt1b (committed, to s600)"}
             for r in bleed_committed]
    curve += [{"step": r["step"], "D": r["cum_disp"], "gm12": r["gm12"],
               "g0": r["g0"], "ce_r": r["ce_r"],
               "cos_delta_fact_g0": r["cos_delta_fact_g0"],
               "cos_delta_fact_m12": r["cos_delta_fact_m12"],
               "provenance": "opt1b2 (this run)"}
              for r in reads]
    curve.sort(key=lambda r: r["step"])
    # dedupe (the committed curve ends at 600; the continuation starts >600)
    _seen, curve_dd = set(), []
    for r in curve:
        if r["step"] not in _seen:
            _seen.add(r["step"])
            curve_dd.append(r)
    curve = curve_dd

    # ---- THE PUMP-CLIFF MONITOR: the per-interval slope d(g-12)/dD ladder
    # along the FULL bleed (monitor only; never adjudicates)
    pump_rows = []
    for a, b in zip(curve, curve[1:]):
        dD = b["D"] - a["D"]
        if dD <= 1e-12:
            continue
        pump_rows.append({"step_from": a["step"], "step_to": b["step"],
                          "D_from": a["D"], "D_to": b["D"],
                          "D_mid": 0.5 * (a["D"] + b["D"]),
                          "slope_gm12_per_D": (b["gm12"] - a["gm12"]) / dD})
    early = [abs(p["slope_gm12_per_D"]) for p in pump_rows
             if p["D_mid"] < 2.0]
    early_max = max(early) if early else None
    accel_rows = []
    if early_max:
        accel_rows = [p for p in pump_rows
                      if p["D_mid"] >= 2.0
                      and abs(p["slope_gm12_per_D"]) > 2.0 * early_max]
    pump_cliff_monitor = {
        "rows": pump_rows,
        "early_max_abs_slope_D_lt_2": early_max,
        "first_accelerated_row": (accel_rows[0] if accel_rows else None),
        "acceleration_criterion": "|slope| > 2x the max |slope| at D<2.0 "
                                  "(the early bleed texture) — where does "
                                  "the decay ACCELERATE?",
        "note": "THE PUMP-CLIFF MONITOR (opt1c's signature: pump to 0.955 "
                "at D~0.33 then cliff by D~1.0 on the big-step path): "
                "d(g-12)/dD per inter-read interval along the FULL bleed "
                "(committed + continuation). Monitor only — never "
                "adjudicates.",
    }

    # =====================================================================
    # ADJUDICATION (registered clauses; composite order frozen:
    # BLEED-KILLS-AT-ADAM-GATE -> BLEED-SPARED-PAST-GATE -> CAP-NEITHER;
    # precedence: the kill wins iff its t_x precedes the measured
    # 2.6-crossing step; kill first at a tie step; no shopping)
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
                                         "gm12_bracket": list(
                                             stop["gm12_bracket"]),
                                         "D_bracket": list(stop["D_bracket"])},
            "BLEED_SPARED_PAST_GATE": {"fires": False},
            "CAP_NEITHER": {"fires": False},
        }
        if in_window:
            verdict = "BLEED-KILLS-AT-ADAM-GATE"
            clause = (f"the bleed died at interpolated displacement D_kill "
                      f"{d_kill:.3f} (t_x {stop['t_x']:.1f}, bracket "
                      f"{stop['bracket']}, g-12 bracket "
                      f"{stop['gm12_bracket']}) — INSIDE the registered "
                      f"window [{kill_window_lo}, {kill_window_hi}]: the "
                      f"re-orienting path reaches the same gate as the "
                      f"guillotine; 'the gate is raw displacement' "
                      f"survives cross-class (reunifying e188 within AND "
                      f"across classes at the gate magnitude).")
        else:
            verdict = "KILL-OUT-OF-WINDOW (no bar)"
            clause = (f"the bleed died at interpolated displacement D_kill "
                      f"{d_kill:.3f} (t_x {stop['t_x']:.1f}) — OUTSIDE the "
                      f"registered window [{kill_window_lo}, "
                      f"{kill_window_hi}]: NEITHER bar fires (opt1b's "
                      f"convention: a kill outside the window is a graded "
                      f"outcome; the full fact-vs-D curve is reported "
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
            "CAP_NEITHER": {"fires": False},
        }
        verdict = "BLEED-SPARED-PAST-GATE (post-gate kill reported)"
        clause = (f"the bleed PASSED D = 2.6 alive (g-12 "
                  f"{d26.get('gm12', float('nan')):.4f} > {SHUT_BAR} at D "
                  f"{d26.get('cum_disp', float('nan')):.4f}, step "
                  f"{d26.get('step', '?')}) — the gate is trajectory-class-"
                  f"typed all the way; the law's final form names the "
                  f"classes (guillotine ~2.5 / annihilation ~0.9 / bleed "
                  f"its own: measured kill-D "
                  f"{stop['D_kill_interp']:.3f} at t_x {stop['t_x']:.1f}, "
                  f"bracket {stop['bracket']}).")
    elif stop["kind"] == "spared":
        d26 = stop.get("d26_row") or {}
        bars = {
            "BLEED_KILLS_AT_ADAM_GATE": {"fires": False},
            "BLEED_SPARED_PAST_GATE": {"fires": True,
                                       "d26_row": d26,
                                       "gm12_at_d33": stop["gm12_at_d33"],
                                       "D_at_d33": stop["D_at_d33"],
                                       "step_at_d33": stop["step"]},
            "CAP_NEITHER": {"fires": False},
        }
        verdict = "BLEED-SPARED-PAST-GATE"
        clause = (f"the bleed PASSED D = 2.6 alive (g-12 "
                  f"{d26.get('gm12', float('nan')):.4f} > {SHUT_BAR} at D "
                  f"{d26.get('cum_disp', float('nan')):.4f}, step "
                  f"{d26.get('step', '?')}) and stayed alive through D = "
                  f"{D_STOP} (g-12 {stop['gm12_at_d33']:.4f} at D "
                  f"{stop['D_at_d33']:.4f}, step {stop['step']}; g0 "
                  f"{stop['g0_at_d33']:.4f}, CE_R "
                  f"{stop['ce_r_at_d33']:.4f}) — the gate is trajectory-"
                  f"class-typed all the way; the law's final form names "
                  f"the classes (guillotine ~2.5 / annihilation ~0.9 / "
                  f"bleed its own: UNREACHED within this run's window — "
                  f"alive at the 3.3 alive-stop; the opt1b projection "
                  f"named ~step 1829 / D ~10, EXTRAPOLATED, never "
                  f"adjudicated here).")
    else:
        fin = journal[-1] if journal else j600
        tail = [r["step_disp"] for r in journal[-40:]] or \
               [float(j600["step_disp"])]
        rate = sum(tail) / max(len(tail), 1)
        d_now = fin["cum_disp"]
        # PROJECTED crossings (labeled EXTRAPOLATED; never adjudicate)
        proj_26 = ((D_SPARE - d_now) / rate if d_now < D_SPARE
                   and rate > 0 else None)
        proj_33 = ((D_STOP - d_now) / rate if d_now < D_STOP
                   and rate > 0 else None)
        g_proj = None
        if len(reads) >= 2 and reads[-1]["gm12"] < reads[-2]["gm12"]:
            s0, s1 = reads[-2]["step"], reads[-1]["step"]
            v0, v1 = reads[-2]["gm12"], reads[-1]["gm12"]
            if v0 > v1:
                g_proj = s1 + (v1 - SHUT_BAR) / (v0 - v1) * (s1 - s0)
        bars = {
            "BLEED_KILLS_AT_ADAM_GATE": {"fires": False},
            "BLEED_SPARED_PAST_GATE": {"fires": False},
            "CAP_NEITHER": {"fires": True},
            "projection": {"steady_disp_rate": rate,
                           "extrapolated_steps_to_D_2.6": (
                               fin["step"] + proj_26
                               if proj_26 is not None else None),
                           "extrapolated_steps_to_D_3.3": (
                               fin["step"] + proj_33
                               if proj_33 is not None else None),
                           "extrapolated_kill_step": g_proj,
                           "label": "EXTRAPOLATED — projections never "
                                    "adjudicate"},
        }
        verdict = "CAP-NEITHER"
        clause = (f"neither fired within the {STEP_CAP}-step cap (final "
                  f"|d| {d_now:.4f} at step {fin['step']}, last g-12 "
                  f"{(reads[-1]['gm12'] if reads else last_read_gm12):.4f}); "
                  f"the fact-vs-D curve + the labeled projection are "
                  f"reported; no adjudication.")
    log("=" * 78)
    log(f"OPT1B2 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # ---- alignment summary (co-read; report, no adjudication)
    my_reads = [r for r in reads]
    align_summary = None
    if my_reads:
        earlyc = [r["cos_delta_fact_g0"] for r in my_reads
                  if r["cum_disp"] <= 1.0]
        latec = [r["cos_delta_fact_g0"] for r in my_reads
                 if r["cum_disp"] > 1.0]
        align_summary = {
            "mean_cos_g0_D_le_1": (sum(earlyc) / len(earlyc)) if earlyc
            else None,
            "mean_cos_g0_D_gt_1": (sum(latec) / len(latec)) if latec
            else None,
            "min_cos_g0": min(r["cos_delta_fact_g0"] for r in my_reads),
            "max_cos_g0": max(r["cos_delta_fact_g0"] for r in my_reads),
            "note": "the W022/W023 question along the continuation: does "
                    "the cumulative displacement drift more (more "
                    "negative) or less death-directed as D grows past the "
                    "opt1b window? opt1b's bleed sat at -0.025..-0.041 "
                    "(slightly LESS death-directed as D grew); opt1's "
                    "arms at -0.015..-0.105; W022's wash read ~-0.44",
        }

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "opt1b2_crossing",
        "date": common.now_iso(),
        "status": "COMPLETE — adjudicated (this write replaces all PARTIAL "
                  "progressive writes)",
        "registration": ("the dispatch's registration IS the registration "
                         "(the three bars quoted verbatim in the module "
                         "docstring and in registered_prediction, frozen "
                         "before compute)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("where does the re-orienting path ACTUALLY die? The "
                     "Adam gate is 2.49-2.84; the annihilation is 0.920 "
                     "(opt1c); opt1b's labeled projection said D=2.6 at "
                     "~step 761 alive and kill ~step 1829 (D ~10). THE "
                     "GATE-CROSSING READ: continue the committed s600 "
                     "bleed to the kill / D=3.3 alive / the 1200-step cap, "
                     "reading finely around the D=2.6 crossing"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "e185 arm C (CONTROL) / opt1 A2b / opt1b "
                             "VERBATIM — the consolidated host fact under "
                             "its neutral/corpus wash",
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
            "read_cadence": {"fine_zone": list(FINE_ZONE),
                             "fine_gap": FINE_GAP,
                             "standard_gaps": {"hi": GAP_HI, "mid": GAP_MID,
                                               "lo": GAP_LO,
                                               "triggers": [0.55, 0.35]},
                             "forced_reads": ["D=2.6 crossing",
                                              "D=3.3 crossing", "cap"]},
            "measure_light": "g-12 / g0 batteries + CE_R + alignment + "
                             "displacement + pre-clip grad norms + the "
                             "pump-cliff slope monitor (the dispatch's "
                             "read list; no census/deletions)",
        },
        "resume": {
            "method": ("CONTINUE from opt1b's committed s600 state: "
                       "runs/opt1b/chunk_state.pt (model + generator "
                       "state + step 600 + the 600-row journal + the 14 "
                       "reads — the state a cross-process resumer loads), "
                       "cross-gated bit vs runs/checkpoints/"
                       "opt1b_a2b_sgd_1e-2_s600.pt; plain SGD m=0/wd=0 "
                       "is STATELESS (no optimizer state to carry); the "
                       "stream position carries in the generator state — "
                       "the input stream CONTINUES bit-identically from "
                       "step 601, never repeats (union md5 uniqueness "
                       "gated)"),
            "parent_journal": {"path": "runs/opt1b/journal.jsonl",
                               "rows": len(j600_file),
                               "md5": parent_journal_md5},
            "parent_state_hash_chain_tail": chain_tail,
            "state_hash_chain_this_run": [c["flat_md5_at_end"]
                                          for c in chunks_prov],
            "gates": "see gates.G_RESUME (the resume bit-consistency "
                     "gate: md5 chain + s600.pt bit + journal row-equal "
                     "+ hard-bound step-600 row/read + displacement "
                     "recompute + committed 1..10 x_md5s)",
        },
        "arm": {
            "tag": "a2b_sgd_1e-2 (continued past the s600 cap)",
            "parent": "runs/opt1b/metrics.json arms.a2b_sgd_1e-2 (600 "
                      "steps, CAP-NEITHER, alive 0.601 at D 1.4338; "
                      "projection D=2.6 ~step 761 ALIVE / kill ~step 1829 "
                      "— EXTRAPOLATED, never adjudicated)",
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
                  "G_INPUTS": G_INPUTS, "G_DRAWFREE": G_DRAWFREE,
                  "G_CHUNKS": G_CHUNKS},
        "references": {
            "opt1": {"metrics": "runs/opt1/metrics.json",
                     "role": "the Adam arms' fact-vs-D curves (the "
                             "overlay, plotted from committed data, never "
                             "rerun); D_kill 2.489 and the gate band "
                             "2.49-2.84 (T139)"},
            "opt1b": {"metrics": "runs/opt1b/metrics.json",
                      "role": "the bleed's committed first 600 steps (the "
                              "curve + the chunk_state + the journal this "
                              "run resumes from); T142's CAP-NEITHER"},
            "opt1c": {"metrics": "runs/opt1c/metrics.json",
                      "role": "the annihilation arm (raw-gradient "
                              "direction at Adam's size: D_kill 0.920; "
                              "its curve + along-path densification are "
                              "the overlay's third class); T143's "
                              "KILL-OUT-OF-WINDOW"},
            "e185": {"metrics": "runs/e185/metrics.json",
                     "role": "the licensed cell + the stored input md5s "
                             "the committed stream re-derives at steps "
                             "1..10"},
        },
        "fact_vs_D_curve_this_run": [{"step": r["step"], "D": r["cum_disp"],
                                      "gm12": r["gm12"], "g0": r["g0"],
                                      "ce_r": r["ce_r"],
                                      "cos_delta_fact_g0":
                                          r["cos_delta_fact_g0"],
                                      "cos_delta_fact_m12":
                                          r["cos_delta_fact_m12"],
                                      "forced": r["forced"]}
                                     for r in reads],
        "bleed_full_curve": curve,
        "bleed_committed_overlay": [{"step": r["step"], "D": r["D"],
                                     "gm12": r["gm12"]}
                                    for r in bleed_committed],
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
            "alignment": align_summary,
            "ce_r": {"note": "does the organism keep learning? CE_R at "
                             "every checkpoint (rows in "
                             "fact_vs_D_curve_this_run / "
                             "bleed_full_curve)",
                     "root": root_cells["ce_r"],
                     "s600_committed": r600["ce_r"],
                     "final": reads[-1]["ce_r"] if reads else None},
            "pump_cliff_monitor": pump_cliff_monitor,
        },
        "adjudication": {
            "bars": bars,
            "verdict": verdict, "clause": clause,
            "stop": {k: v for k, v in stop.items()},
            "constants": {"D_spare_gate": D_SPARE, "D_stop": D_STOP,
                          "kill_window": list(KILL_WINDOW),
                          "adam_gate_band": list(ADAM_GATE_BAND),
                          "shut_bar": SHUT_BAR, "step_cap": STEP_CAP,
                          "D_kill_adam_A0": D_KILL_ADAM,
                          "D_kill_opt1c_annihilation": OPT1C_DKILL},
            "precedence": ("the kill wins iff its t_x precedes the "
                           "measured 2.6-crossing step (a dead read AT "
                           "the crossing wins the tie); an alive 2.6 "
                           "crossing certifies BLEED-SPARED-PAST-GATE and "
                           "the run continues per the stop rule — a "
                           "post-gate kill is reported as the bleed's own "
                           "measured kill-D, never re-adjudicated"),
            "composite_order": "BLEED-KILLS-AT-ADAM-GATE -> "
                               "BLEED-SPARED-PAST-GATE -> CAP-NEITHER "
                               "(frozen before compute)",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1, single seed lineage (10902, A2b's; the "
                            "replicate ladder is a separate dispatch "
                            "decision); ONE root, ONE input stream (the "
                            "committed one, resume-gated bit); single "
                            "family (the e131 consolidated line); the "
                            "continuation differs from opt1b's bleed in "
                            "NOTHING — same optimizer, same stream, same "
                            "clip — it is the same training, resumed "
                            "past its cap"),
            "extrapolation_free": ("every adjudication input is MEASURED "
                                   "inside this run's window: the kill "
                                   "(if any) is bracketed by reads and "
                                   "interpolated linearly-in-step (the "
                                   "fine cadence bounds the bracket at 10 "
                                   "steps inside the crossing zone); the "
                                   "spare is certified by FORCED reads at "
                                   "the measured 2.6- and 3.3-crossing "
                                   "steps; only CAP-NEITHER carries "
                                   "projections, labeled EXTRAPOLATED "
                                   "and non-adjudicating by registration"),
            "float_texture": ("CPU fp32, this process, 4 threads (the "
                              "e185/opt1 reduction order); the resume "
                              "gate bounds cross-process float drift "
                              "explicitly (md5 chain + bit comparisons + "
                              "the displacement recompute, observed "
                              + ("bit" if disp_recheck_diff < G_BIT_TOL
                                 else "within tolerance") + ")"),
            "committed_data_reuse": ("opt1's Adam arms, opt1c's "
                                     "annihilation curve + densification, "
                                     "and opt1b's committed first-600 "
                                     "bleed curve are REUSED from "
                                     "committed metrics (plotted, never "
                                     "rerun); this run's reads begin at "
                                     "step 640 + the forced gate reads"),
            "no_guarantees": ("nothing was guaranteed ex ante: the bleed "
                              "could die at 2.5, pass the gate alive and "
                              "die at 3.0, reach 3.3 alive, or creep to "
                              "the cap — the openness is the point; the "
                              f"stop actually observed is '{stop['kind']}'"
                              + (" (post-gate)" if stop.get("post_gate")
                                 else "")),
            "interpolation_texture": ("the kill's t_x/D_kill are "
                                      "linear-in-step interpolations "
                                      "inside a measured bracket whose "
                                      "width is reported; the bracket is "
                                      "the honest resolution limit"),
            "pump_cliff_monitor_is_a_monitor": ("the slope ladder "
                                                "reports WHERE the decay "
                                                "accelerates; it gates "
                                                "nothing and adjudicates "
                                                "nothing"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": {
            fin_name: {"path": f"runs/checkpoints/{fin_name}.pt",
                       "arm": "a2b_sgd_1e-2", "steps": int(fin_step),
                       "stop": stop["kind"]},
            "chunk_state": {"path": "runs/opt1b2/chunk_state.pt",
                            "note": "the resumable continuation state "
                                    "(model + generator + step + "
                                    "continuation journal + reads)"},
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

    plot_fact_vs_D(rd / "opt1b2_fact_vs_D.png", curve, adam_arms,
                   anni_curve, anni_densify, reads, verdict, clause, stop)
    plot_coreads(rd / "opt1b2_coreads.png", curve, journal, reads,
                 chunks_prov, stop, pump_rows)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'opt1b2_fact_vs_D.png'}, "
        f"{rd / 'opt1b2_coreads.png'}, journal.jsonl, chunk_state.pt, "
        f"runs/checkpoints/{fin_name}.pt")
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


def plot_fact_vs_D(path, curve, adam_arms, anni_curve, anni_densify,
                   my_reads, verdict, clause, stop):
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.2),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    # the bleed: committed (opt1+opt1b) + this continuation
    com = [c for c in curve if c["provenance"].startswith("opt1b (")]
    new = [c for c in curve if c["provenance"].startswith("opt1b2")]
    ax.plot([c["D"] for c in com], [c["gm12"] for c in com], "o-", ms=4.5,
            lw=1.8, color="royalblue", alpha=0.9,
            label="THE BLEED SGD 1e-2 — committed (opt1 steps 0..40 + "
                  "opt1b to s600)")
    if new:
        ax.plot([c["D"] for c in new], [c["gm12"] for c in new], "D-", ms=6,
                lw=2.6, color="blue", alpha=0.95, zorder=5,
                label="THE BLEED — opt1b2 continuation (this run)")
    # the Adam arms (opt1 committed; displacement keyed "cum_disp")
    for t, rows in adam_arms.items():
        col, lbl = ADAM_STYLE[t]
        pts = [(r["cum_disp"], r["gm12"]) for r in rows if r["step"] > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--", ms=5,
                lw=1.5, color=col, alpha=0.85, label=lbl)
    # the annihilation arm (opt1c) + its along-path densification
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
    ax.axvspan(FINE_ZONE[0], FINE_ZONE[1], color="royalblue", alpha=0.06)
    ax.axvline(D_SPARE, ls=":", lw=2.0, color="navy", alpha=0.8,
               label="D=2.6 spared gate")
    ax.axvline(D_STOP, ls=":", lw=1.4, color="blue", alpha=0.6,
               label="D=3.3 alive-stop (window top + margin)")
    for yv, col, lbl in ((0.50, "seagreen", "0.50 SPARE"),
                         (SHUT_BAR, "tab:purple", f"{SHUT_BAR} DISSOLVE")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    if stop["kind"] == "kill":
        ax.plot([stop["D_kill_interp"]], [SHUT_BAR], "*", ms=19,
                color="yellow", mec="k", zorder=6,
                label=f"bleed kill (interp D {stop['D_kill_interp']:.3f}"
                      + (", POST-gate" if stop.get("post_gate")
                         else ", pre-gate") + ")")
    if stop["kind"] == "spared":
        d26 = stop.get("d26_row")
        if d26:
            ax.plot([d26["cum_disp"]], [d26["gm12"]], "*", ms=19,
                    color="lime", mec="k", zorder=6,
                    label=f"passed D=2.6 ALIVE (g-12 {d26['gm12']:.3f})")
        ax.plot([stop["D_at_d33"]], [stop["gm12_at_d33"]], "*", ms=15,
                color="lime", mec="k", alpha=0.6, zorder=6,
                label=f"alive at D=3.3 (g-12 {stop['gm12_at_d33']:.3f})")
    if stop["kind"] == "kill" and stop.get("d26_row"):
        d26 = stop["d26_row"]
        ax.plot([d26["cum_disp"]], [d26["gm12"]], "*", ms=15, color="lime",
                mec="k", alpha=0.7, zorder=6,
                label=f"passed D=2.6 ALIVE (g-12 {d26['gm12']:.3f})")
    ax.set_xlabel(r"cumulative displacement $\|\theta_t-\theta_0\|_2$ "
                  "(2.74M params)")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.08, 5.3)
    ax.legend(fontsize=6.9, loc="lower left")
    ax.set_title("THE GATE-CROSSING READ — the bleed continued to the "
                 "crossing: FOUR trajectory classes at their own gates "
                 "(committed parents + this run)", fontsize=9.5)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"OPT1B2 VERDICT: {verdict}", fontsize=10.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.05
    for wd in textwrap.wrap(clause, width=86, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=7.2, va="top",
                family="monospace")
        y -= 0.026
    y -= 0.02
    ax.text(0.02, y, "  step     D        g-12    g0     CE_R    cos(g0)  "
                     " forced", fontsize=7.2, va="top", family="monospace")
    y -= 0.028
    for r in my_reads:
        cos = r["cos_delta_fact_g0"]
        ax.text(0.02, y,
                f"  {r['step']:>5d}  {r['cum_disp']:7.4f}  {r['gm12']:.4f}"
                f"  {r['g0']:.4f}  {r['ce_r']:.4f}  "
                + (f"{cos:+.4f}" if cos is not None else "    n/a")
                + f"  {r['forced'] or ''}",
                fontsize=6.9, va="top", family="monospace",
                color=("blue" if r["forced"] else "black"))
        y -= 0.026
    fig.suptitle("OPT1B2 — THE GATE-CROSSING READ: where does the "
                 "re-orienting path actually die?", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coreads(path, curve, journal, reads, chunks_prov, stop, pump_rows):
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    # (0,0) D(t) vs step + the gates
    ax = axes[0, 0]
    ax.plot([r["step"] for r in journal],
            [r["cum_disp"] for r in journal], "-", lw=1.8,
            color="royalblue", label="cumulative |d| (continuation)")
    ax.axhline(D_SPARE, ls=":", lw=1.8, color="navy", alpha=0.9,
               label="D=2.6 spared gate")
    ax.axhspan(ADAM_GATE_BAND[0], ADAM_GATE_BAND[1], color="tab:purple",
               alpha=0.10, label="Adam kill band 2.49-2.84 (opt1)")
    ax.axhline(D_KILL_ADAM, ls="--", lw=1.0, color="crimson", alpha=0.7,
               label="A0 D_kill 2.489")
    ax.axhline(OPT1C_DKILL, ls="--", lw=1.0, color="black", alpha=0.7,
               label=f"opt1c annihilation D_kill {OPT1C_DKILL:.3f}")
    ax.axhline(D_STOP, ls=":", lw=1.4, color="blue", alpha=0.6,
               label="D=3.3 alive-stop")
    for c in chunks_prov[:-1]:
        ax.axvline(c["step_to"], ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.set_xlabel("wash step (continuation 601..; chunk boundaries dotted)")
    ax.set_ylabel(r"$\|\theta_t-\theta_0\|_2$")
    ax.legend(fontsize=7.0, loc="lower right")
    ax.set_title("D(t) — the measured trajectory vs the three gates",
                 fontsize=10)

    # (0,1) g-12 vs step along the FULL bleed
    ax = axes[0, 1]
    ax.plot([c["step"] for c in curve], [c["gm12"] for c in curve], "o-",
            ms=4.5, lw=1.8, color="royalblue",
            label="g-12 (committed 0..600 + this run)")
    ax.plot([r["step"] for r in reads], [r["gm12"] for r in reads], "D",
            ms=7, color="blue", label="opt1b2 reads")
    ax.axhline(SHUT_BAR, ls="--", lw=1.2, color="tab:purple",
               label=f"{SHUT_BAR} DISSOLVE")
    if stop["kind"] in ("kill", "spared") and stop.get("d26_row"):
        ax.axvline(stop["d26_row"]["step"], ls=":", lw=1.6, color="navy",
                   alpha=0.8, label=f"D=2.6 crossing @ s"
                                    f"{stop['d26_row']['step']}")
    if stop["kind"] == "kill":
        ax.axvline(stop["step"], ls="-", lw=1.4, color="tab:purple",
                   alpha=0.6, label=f"kill @ s{stop['step']}")
    ax.set_xlabel("wash step")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("the kill clock in STEPS along the full bleed",
                 fontsize=10)

    # (0,2) CE_R vs D (organism health)
    ax = axes[0, 2]
    ax.plot([c["D"] for c in curve], [c["ce_r"] for c in curve], "o-",
            ms=4.5, lw=1.8, color="saddlebrown", label="CE_R (organism)")
    ax.axhline(curve[0]["ce_r"], ls="--", lw=1.0, color="gray",
               alpha=0.8, label=f"root CE_R {curve[0]['ce_r']:.3f}")
    ax.axhline(4.788, ls=":", lw=1.2, color="black", alpha=0.7,
               label="opt1c annihilation CE_R 4.79 (organism devastated)")
    ax.set_xlabel("cumulative displacement D")
    ax.set_ylabel("CE_R")
    ax.legend(fontsize=7.0, loc="upper left")
    ax.set_title("does the organism keep learning under the bleed?",
                 fontsize=10)

    # (1,0) alignment vs D (the W022/W023 co-read)
    ax = axes[1, 0]
    rows = [c for c in curve if c["cos_delta_fact_g0"] is not None]
    ax.plot([c["D"] for c in rows], [c["cos_delta_fact_g0"] for c in rows],
            "o-", ms=4.5, lw=1.8, color="teal",
            label=r"cos($\Delta\theta$, $\nabla$g0)")
    ax.plot([c["D"] for c in rows],
            [c["cos_delta_fact_m12"] for c in rows], "s--", ms=4,
            lw=1.5, color="olive", alpha=0.8,
            label=r"cos($\Delta\theta$, $\nabla$m12)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(-0.44, ls="--", lw=1.0, color="gray", alpha=0.8,
               label="W022 wash read ~-0.44")
    ax.set_xlabel("cumulative displacement D")
    ax.set_ylabel("alignment cos")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("ALIGNMENT — does the path drift death-directed as D "
                 "grows? (neg = death-aligned)", fontsize=9.5)

    # (1,1) THE PUMP-CLIFF MONITOR: d(g-12)/dD vs D_mid
    ax = axes[1, 1]
    ax.plot([p["D_mid"] for p in pump_rows],
            [p["slope_gm12_per_D"] for p in pump_rows], "o-", ms=5,
            lw=1.8, color="mediumvioletred")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvspan(FINE_ZONE[0], FINE_ZONE[1], color="royalblue", alpha=0.06)
    ax.axvline(D_SPARE, ls=":", lw=1.6, color="navy", alpha=0.8)
    ax.axvline(D_STOP, ls=":", lw=1.2, color="blue", alpha=0.6)
    ax.set_xlabel("D at interval midpoint")
    ax.set_ylabel(r"d(g-12)/dD  (per unit displacement)")
    ax.set_title("THE PUMP-CLIFF MONITOR — where does the decay "
                 "ACCELERATE along the bleed? (monitor only)", fontsize=9.5)

    # (1,2) per-step texture: step |d| and pre-clip ||g||
    ax = axes[1, 2]
    ax.plot([r["step"] for r in journal], [r["step_disp"] for r in journal],
            "-", lw=1.4, color="royalblue", label="per-step |d|")
    ax.plot([r["step"] for r in journal],
            [r["preclip_gnorm"] * LR_A2B for r in journal], "-", lw=1.2,
            color="darkorange", alpha=0.8,
            label="lr x pre-clip ||g|| (the SGD step before clip rescale)")
    ax.set_xlabel("wash step (continuation)")
    ax.set_ylabel("L2 per step")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("the bleed's per-step texture (re-orientation at work)",
                 fontsize=10)

    fig.suptitle("OPT1B2 — co-reads: D(t), the clock, the organism, the "
                 "alignment, the pump-cliff monitor, the per-step texture",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
