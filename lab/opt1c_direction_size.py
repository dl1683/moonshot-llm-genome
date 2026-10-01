"""OPT1C — THE DIRECTION-SIZE FACTORIAL (R58-critic's forced cell; T142's
adjudicator; the third cell of the opt1 arc).

WHY: opt1 (runs/opt1/metrics.json, T139) decomposed the wash kill into
CLOCK (Adam's sign-normalization, 1683x/step at matched lr) and GATE
(displacement ~2.49-2.84 across Adam variants). opt1b (runs/opt1b/
metrics.json, T142) ran the direct raw-gradient path (SGD lr 1e-2) to
its frozen 600-step cap: CAP-NEITHER — alive 0.601 at D 1.43, the
D=2.6 gate unreached, the (never-adjudicated) projection reading the
kill near D~10. THE OPEN QUESTION: inside the Adam kill, is it the
raw-gradient DIRECTION at small size that spares (SIZE carries the
kill), or would ANY direction at Adam's step size kill (cumulative
displacement is direction-robust)? opt1c splits direction from size
in ONE arm: the raw batch-gradient DIRECTION at Adam's measured
per-step L2 — "SGD's direction at Adam's size."

THE CELL: the licensed e185 wash cell VERBATIM (opt1's machinery:
root runs/checkpoints/e131_consolidated_e113.pt, stream seed 10902,
bit-identical batches — 16 e170 neutral-bank anchors + 16 random
corpus windows, TRUE targets, full-token CE, clip 1.0 — A0's
conventions throughout). The ONLY delta is the update rule: at each
step t, with g_t the batch gradient (post-clip; the clip is a
positive rescale so the DIRECTION is clip-invariant, pre-clip norm
recorded),

    delta_t = STEP_SCALE * (g_t / ||g_t||_2),   theta_{t} -= delta_t,

i.e. DESCENT along the raw gradient direction (what SGD follows) with
per-step L2 pinned to Adam's measured step size. NOTE THE CORRECTION
vs the critic's letter (r58_critic 1g said "sign-SGD ... ±lr per
coordinate"): this cell runs the L2-NORMALIZED gradient, not
elementwise sign(g) — the dispatch's correction — which (a) is
exactly SGD's direction and (b) holds the per-step L2 EXACTLY at the
measured Adam displacement (sign(g)*lr only approximates it, 1.6550
vs measured 1.6543). STEP_SCALE = 1.6543 (opt1's A0 step-1 L2),
RECOMPUTED at t=0 before the run (one AdamW recipe step from the
root on the step-1 batch, gated bit vs the committed value) and the
MEASURED value is used, documented.

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - KILLS-AT-GATE: "fires if the fact dies at displacement within
    [2.12, 3.27] — cumulative displacement is DIRECTION-ROBUST at
    Adam's size; the alignment integral is epiphenomenal; 'any path
    that reaches the gate kills' holds in its letter."
  - SPARED-AT-GATE: "fires if the trajectory passes D = 2.6 with the
    fact alive (g-12 > 0.27) — the kill is not carried by size alone;
    DIRECTION (or normalization structure) carries it; 'any path
    reaching the gate kills' dies in its current letter and the law
    moves to ruler-aligned-displacement currency (e188's RAW-WINS
    then needs reconciliation — report the tension honestly)."
  - CAP-NEITHER: "neither within the cap — report the curve; no
    projection adjudicates."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses,
they do not move the bars):
  * the fact = g-12 (install-60 battery mean p(Z) at ctx offset -12,
    e185's convention); KILL = g-12 <= 0.27 (the arc's SHUT bar),
    interpolated linearly-in-step between the last alive read and the
    first dead read (opt1b's convention); D_kill = the same-bracket
    interpolation of cumulative displacement. Reads are taken at
    EVERY step (the bracket-tightest cadence; the arm resolves
    within a handful of steps because each step moves ~1.65 L2), and
    the kill bracket is additionally DENSIFIED along the measured
    path: g-12 is read at parameter points theta_{s0} + f*(theta_{s1}
    - theta_{s0}) for f in {0.2, 0.4, 0.6, 0.8} — points ON the same
    piecewise-linear trajectory, not new training steps — refining
    D_kill's bracket for the window adjudication. Both the raw and
    the densified brackets are reported.
  * displacement = cumulative ||theta_t - theta_0||_2 over all
    2,739,072 params (fp32, CPU, measured every step), e185/opt1/
    opt1b's currency verbatim; the per-step displacement is pinned at
    STEP_SCALE by construction (machinery gate: max |step_disp -
    STEP_SCALE| reported).
  * D = 2.6 = the SPARED gate (opt1b's frozen number); D = 3.3 = the
    run's alive-stop (the kill window's top 3.27 + margin) — the run
    continues past a live 2.6 crossing to 3.3 so a kill INSIDE
    [2.12, 3.27] always outranks an earlier live pass (composite
    order KILLS -> SPARED -> CAP, opt1b's order verbatim; at a tie
    step the dead read is processed first).
  * stop = whichever FIRST of: kill (g-12 <= 0.27 at a read), D >=
    3.3 reached alive (forced read at the crossing step), 300-step
    cap. SPARED fires only when the trajectory is measured alive at
    BOTH the 2.6-crossing read AND the 3.3-crossing read; a kill
    outside [2.12, 3.27] fires NO bar (opt1b's convention: graded
    outcome, curve reported verbatim).
  * the arm guarantees NOTHING (stated before compute): it could
    kill at the gate, pass it alive, or bleed like SGD — that
    openness is the point.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE the arm runs): the
standard cell gates (corpus ZEPH count 0; splice mix {FLORIZEL: 19,
ELIZABETH: 41}; battery shapes 60 x (130 +- j); neutral bank 16
starts bit-equal to e185's stored list; root gated vs e151's
before-cells) PLUS the t=0 BIT-IDENTITY GATE: the step-1 batch md5
vs e185's stored hash, the step-1 forward CE and pre-clip grad norm
vs opt1's committed A0 row, and one fresh AdamW-recipe step from the
root reproducing A0's committed step-1 L2 displacement (tol 5e-6 bit
/ 0.05 fallback; abort on control failure — without it the size
match that defines the arm is untrustworthy).

CO-READS: (1) the fact-vs-D curve three-way: opt1's Adam arms +
opt1b's bleed OVERLAID from COMMITTED data (runs/opt1/metrics.json +
runs/opt1b/metrics.json loaded and plotted, never rerun) + this arm;
(2) CE_R at every checkpoint (organism health); (3) alignment
cos(delta_theta_t, grad g0_t) at every checkpoint (cumulative-
displacement convention, opt1 verbatim; co-reported for the m12
ruler; critic's sign: negative = death-aligned); (4) per-step
direction persistence cos(u_t, u_{t-1}) (why D(t) grows sublinearly).

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1
forced before torch; the GPU is g1bS's — never claimed), torch
threads 4 (the e185/opt1 reduction order), ONE arm, ckpt-RESUMABLE
chunks <= 180 s per the opt1b convention (full state — model +
generator + step + journal — round-trips through runs/opt1c/
chunk_state.pt after each chunk; each chunk LOADS the previous
chunk's file; the stream position carries in the generator state;
chunk-boundary parameter md5s form a hash chain), n=1, single seed
lineage (10902).

Outputs: runs/opt1c/{metrics.json, opt1c_fact_vs_D.png,
opt1c_coreads.png, chunk_state.pt, journal.jsonl}; checkpoint
runs/checkpoints/opt1c_sgrad_dir_s<final>.pt. No NOTES/THINKING/
QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python opt1c_direction_size.py    (OPT1C_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185/e187/opt1/opt1b)

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

SMOKE = os.environ.get("OPT1C_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "opt1c is CPU-only by dispatch"

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
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the arm + the run envelope (dispatch-frozen) -------------------------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185/opt1/opt1b)
LR_ADAMW = 1e-3                   # A0's lr (the t=0 reproduction only; this arm has NO lr)
STEP_SCALE_COMMITTED = 1.6542880535125732   # opt1 A0's committed step-1 L2 (the reference)
CE1_COMMITTED = 1.356567621231079           # opt1 A0's committed step-1 batch CE
GN1_COMMITTED = 0.9829167127609253          # opt1 A0's committed step-1 pre-clip grad norm
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
CHUNK_CAP = 178.0                 # dispatch: <=180 s per chunk (margin for step overshoot)
STEP_CAP = 300                    # dispatch: 300-step cap (whichever-first stop rule)
D_SPARE = 2.6                     # the SPARED gate (opt1b's frozen number)
D_STOP = 3.3                      # the alive-stop (kill window top 3.27 + margin)
KILL_WINDOW = (2.12, 3.27)        # 2.49-2.84 +- 15% (the registered letter's window)
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE)
SURVIVE_NOTE = 0.50               # SPARE reference line (opt1's convention; report-only)
DENSIFY_F = (0.2, 0.4, 0.6, 0.8)  # along-path kill-bracket densification fractions
E170_ANCHOR_SEED = 170            # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
ADAM_GATE_BAND = (2.49, 2.84)     # opt1's measured Adam kill displacements (T139)
if SMOKE:                         # true shakedown trims (documented in deviations)
    STEP_CAP, CHUNK_CAP = 4, 12.0

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
    "kills_at_gate": "KILLS-AT-GATE: \"fires if the fact dies at displacement "
        "within [2.12, 3.27] — cumulative displacement is DIRECTION-ROBUST at "
        "Adam's size; the alignment integral is epiphenomenal; 'any path that "
        "reaches the gate kills' holds in its letter.\"",
    "spared_at_gate": "SPARED-AT-GATE: \"fires if the trajectory passes D = "
        "2.6 with the fact alive (g-12 > 0.27) — the kill is not carried by "
        "size alone; DIRECTION (or normalization structure) carries it; 'any "
        "path reaching the gate kills' dies in its current letter and the law "
        "moves to ruler-aligned-displacement currency (e188's RAW-WINS then "
        "needs reconciliation — report the tension honestly).\"",
    "cap_neither": "CAP-NEITHER: \"neither within the cap — report the curve; "
        "no projection adjudicates.\"",
    "operationalizations": "the arm = the licensed e185 wash cell VERBATIM "
        "(root, seed-10902 stream, bit-identical batches, TRUE targets, "
        "full-token CE, clip 1.0) with ONE delta: the update is "
        "delta_t = STEP_SCALE * (g_t/||g_t||_2), theta -= delta_t (DESCENT — "
        "SGD's direction at Adam's measured per-step L2; g_t post-clip, "
        "direction clip-invariant, pre-clip norm recorded); STEP_SCALE = "
        "opt1 A0's step-1 L2 1.6542880535125732, RECOMPUTED at t=0 by one "
        "fresh AdamW-recipe step gated bit vs the committed value, the "
        "MEASURED number used; KILL = g-12 <= 0.27 interpolated linearly-"
        "in-step between the last alive and first dead read (reads EVERY "
        "step) + along-path densification at f in {0.2,0.4,0.6,0.8} "
        "(parameter points ON the same trajectory, not new steps); D_kill = "
        "same-bracket interpolation of cumulative displacement; SPARED "
        "certified by alive FORCED reads at BOTH the measured 2.6-crossing "
        "and 3.3-crossing steps; stop = whichever FIRST of kill / D>=3.3 "
        "alive / 300-step cap; a kill OUTSIDE [2.12, 3.27] fires NO bar "
        "(graded outcome, curve reported verbatim; opt1b's convention); "
        "composite order frozen KILLS-AT-GATE -> SPARED-AT-GATE -> "
        "CAP-NEITHER; at a tie step the dead read is processed first.",
    "registration": "the dispatch's registration IS the registration (the "
        "three bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "THE CORRECTION vs the critic's letter: r58_critic 1g proposed 'sign-SGD "
    "(±lr per coordinate)'; this cell runs the L2-NORMALIZED gradient "
    "g_t/||g_t||_2 scaled to Adam's MEASURED step-1 L2 (the dispatch's "
    "correction) — exactly SGD's direction with the per-step L2 pinned "
    "EXACTLY at the measured Adam displacement (sign(g)*lr approximates it, "
    "1.6550 vs 1.6543); the sign-vs-normalized residual is opt2's SIGN arm, "
    "not this cell's.",
    "Reads at EVERY step (the bracket-tightest cadence). The arm is "
    "expected to resolve within a handful of steps (each step moves ~1.65 "
    "L2, so D=3.3 arrives by step ~2-4); the every-step cadence bounds the "
    "kill bracket at one step and the densification refines it WITHIN the "
    "step. Worst case (a direction stall keeping D below 3.3) is bounded by "
    "the 300-step cap at ~8-10 chunks.",
    "Kill-bracket densification reads g-12 at parameter points "
    "theta_{s0} + f*(theta_{s1}-theta_{s0}) for f in {0.2,0.4,0.6,0.8} — "
    "points ON the same piecewise-linear parameter trajectory (between two "
    "committed training steps), not new training steps and not off-path "
    "probes; it refines D_kill's bracket only. Both the raw and densified "
    "brackets are reported; the window adjudication uses the densified "
    "bracket (frozen here before compute).",
    "Chunk cap 178 s (dispatch <=180 s; the margin absorbs one ~2 s step "
    "overshoot); in-chunk checkpoint reads count toward the cap (opt1's "
    "FT_TIME_CAP convention). Chunk state round-trips through runs/opt1c/"
    "chunk_state.pt exactly per the opt1b convention (model + generator "
    "state + step + journal + reads + flags + last direction vector); each "
    "chunk LOADS the previous chunk's file; chunk-boundary parameter md5s "
    "form a hash chain.",
    "This arm has NO learning rate: the update is size-pinned by "
    "construction (per-step L2 = STEP_SCALE exactly; machinery gate "
    "G_STEPSCALE reports max|step_disp - STEP_SCALE|). The batch CE, "
    "pre-clip grad norms, and input md5s are still recorded per step (the "
    "licensed cell's own reads).",
    "Light reads only (g-12/g0 batteries + CE_R + alignment + displacement "
    "+ pre-clip grad norms + direction persistence) — the dispatch's read "
    "list; no census/deletions. CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before "
    "torch; the GPU is g1bS's, never claimed); threads 4; n=1; single seed "
    "lineage (10902).",
    "Smoke mode trims: cap 4 steps, 12 s chunks, no densification, no "
    "adjudication (verdict stamped SMOKE; nothing adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/opt1_optimizer_controls.py / lab/opt1b_sgd_kill.py VERBATIM
# (whose own provenance is lab/e185_noise_wash.py via e187's verified copies —
# the e176n lineage). Copied rather than imported to own the device policy.

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


def load_flat(net: TinyGPT, flat: torch.Tensor) -> None:
    """Copy a flat vector back into parameters (densification loader)."""
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


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
    """Linear crossing of `bar` inside the (alive -> dead) bracket."""
    return s0 + (v0 - bar) / (v0 - v1) * (s1 - s0)


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("opt1c_smoke" if SMOKE else "opt1c")
    chunk_path = rd / "chunk_state.pt"
    log(f"OPT1C THE DIRECTION-SIZE FACTORIAL (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), ONE arm, "
        f"chunks <= {CHUNK_CAP + 2:.0f}s (ckpt-resumable), cap {STEP_CAP} "
        f"steps, n=1, seed lineage {FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (OPT1_METRICS, OPT1B_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    opt1bm = json.loads(OPT1B_METRICS.read_text(encoding="utf-8"))
    a0 = opt1m["arms"]["a0_adamw_ref"]
    a0_s1 = next(r for r in a0["traj"] if r["step"] == 1)
    # hard-bind the t=0 gate references to the COMMITTED values at runtime
    assert abs(a0_s1["cum_disp"] - STEP_SCALE_COMMITTED) < 1e-12, \
        "opt1's committed A0 step-1 displacement drifted vs this file's copy"
    assert abs(a0_s1["ce_batch"] - CE1_COMMITTED) < 1e-12
    assert abs(a0_s1["preclip_gnorm"] - GN1_COMMITTED) < 1e-12
    adam_arms = {t: opt1m["arms"][t]["ckpt_table"]
                 for t in ("a0_adamw_ref", "a3_adamw_warmup30",
                           "a4_adamw_b2_0.999", "a5_adamw_moment_reset")}
    bleed_curve = opt1bm["fact_vs_D_curve"]          # the a2b bleed (committed)
    log(f"parents: opt1 {opt1m['adjudication']['verdict']} (A0 t_x "
        f"{a0['tstar']['t_x']:.2f}, D_kill {D_KILL_ADAM:.4f}); opt1b "
        f"{opt1bm['adjudication']['verdict']} — committed curves loaded "
        f"({sum(len(v) for v in adam_arms.values())} Adam rows, "
        f"{len(bleed_curve)} bleed rows); neither is rerun")

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

    def draw_step1_batch():
        """The step-1 batch, bit-identical to the licensed stream (a fresh
        Generator(10902) with opt1's aj-then-rj draw order)."""
        g = torch.Generator().manual_seed(FREEZE_SEED)
        aj = torch.randint(n_anc, (ANCH_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

    # =====================================================================
    # THE T=0 BIT-IDENTITY GATE + THE MEASURED STEP SCALE (Rule 12)
    # one AdamW-recipe step from the root on the step-1 batch must
    # reproduce opt1 A0's committed step-1 displacement/CE/grad-norm
    # =====================================================================
    x1, y1 = draw_step1_batch()
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    tw = copy.deepcopy(net0)
    tw.train()
    optw = torch.optim.AdamW(tw.parameters(), lr=LR_ADAMW, betas=(0.9, 0.95),
                             weight_decay=0.1)
    logits, _ = tw(x1)
    ce1 = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y1.reshape(-1)).item())
    optw.zero_grad(set_to_none=True)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y1.reshape(-1)).backward()
    gn1 = float(torch.nn.utils.clip_grad_norm_(tw.parameters(), 1.0))
    optw.step()
    disp1 = float(torch.norm(flat_params(tw) - theta0))
    del tw, optw, logits
    d_ce, d_gn, d_disp = (abs(ce1 - CE1_COMMITTED), abs(gn1 - GN1_COMMITTED),
                          abs(disp1 - STEP_SCALE_COMMITTED))
    G_T0 = {
        "step1_x_md5": x1_md5, "step1_x_md5_match_e185": bool(
            x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1, "ce_batch_committed": CE1_COMMITTED,
        "preclip_gnorm_measured": gn1, "preclip_gnorm_committed": GN1_COMMITTED,
        "adamw_step1_L2_measured": disp1,
        "adamw_step1_L2_committed": STEP_SCALE_COMMITTED,
        "diffs": {"ce": d_ce, "gnorm": d_gn, "disp": d_disp},
        "max_abs_diff": max(d_ce, d_gn, d_disp),
        "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
        "bit": bool(max(d_ce, d_gn, d_disp) < G_BIT_TOL),
        "pass": bool(x1_md5 == E185_XHASH[1]
                     and max(d_ce, d_gn, d_disp) < G_FALLBACK_TOL),
        "note": "PRE-DISPATCH CHECK (Rule 12): the step-1 batch md5 vs "
                "e185's stored hash; the forward CE and pre-clip grad norm "
                "vs opt1 A0's committed row; ONE fresh AdamW-recipe step "
                "from the root reproducing A0's committed step-1 L2 — the "
                "size match that defines this arm, gated before the run",
    }
    log(f"G_T0 (t=0 bit-identity vs opt1 A0 committed): x_md5 "
        f"{'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}; "
        f"|dCE| {d_ce:.2e} |dgn| {d_gn:.2e} |dDisp| {d_disp:.2e}: "
        + ("PASS" if G_T0["pass"] else "FAIL")
        + (" (bit)" if G_T0["bit"] else ""))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 bit-identity gate FAILED — the step scale "
                           "is untrustworthy; abort (control failure)")
    STEP_SCALE = float(disp1)          # the MEASURED value (documented)
    log(f"STEP_SCALE = {STEP_SCALE:.16f} (measured; committed "
        f"{STEP_SCALE_COMMITTED:.16f}); effective step-1 SGD lr equivalent "
        f"= STEP_SCALE/||g_1|| = {STEP_SCALE / gn1:.4f}")
    log("WHAT THIS ARM GUARANTEES: NOTHING — it could kill at the gate, "
        "pass it alive, or bleed like SGD; that openness is the point.")

    # =====================================================================
    # THE ARM (chunked, ckpt-resumable; reads at EVERY step)
    # =====================================================================
    journal: list[dict] = []
    reads: list[dict] = []
    densify_rows: list[dict] = []
    chunks_prov: list[dict] = []
    md5_vs_e185: dict[int, bool] = {}
    zeph_checks = 0
    stop: dict | None = None
    d26_crossed = False
    d33_crossed = False
    d26_row: dict | None = None
    G_CHUNKS = {"round_trips": [], "pass": None}
    step_disp_devs: list[float] = []

    net = copy.deepcopy(net0)
    net.train()
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    evl = copy.deepcopy(net0)
    evl.eval()
    prev = theta0.clone()
    prev_u: torch.Tensor | None = None
    step = 0
    chunk_idx = 0

    # the root row is the step-0 read (the kill-interpolation anchor)
    reads.append({"step": 0, "chunk": 0, "forced": None,
                  "gm12": root_cells["gm12"], "g0": root_cells["g0"],
                  "frac_argmax_z": None, "ce_r": root_cells["ce_r"],
                  "cum_disp": 0.0, "cos_delta_fact_g0": None,
                  "cos_delta_fact_m12": None})

    while stop is None:
        # ---- chunk begin: LOAD the previous chunk's state file (the
        # cross-process resume path; chunk 0 starts fresh from the root)
        if chunk_path.exists():
            st = torch.load(chunk_path, map_location="cpu",
                            weights_only=False)
            net.load_state_dict(st["model"])
            net.train()
            gen.set_state(st["gen_state"])
            step = int(st["step"])
            journal = st["journal"]
            reads = st["reads"]
            densify_rows = st["densify_rows"]
            d26_crossed = bool(st["d26_crossed"])
            d33_crossed = bool(st["d33_crossed"])
            prev_u = st["prev_u"]
            d26_row = st["d26_row"]
            md5_vs_e185.update(st["md5_vs_e185"])
            zeph_checks += int(st["zeph_checks"])
            step_disp_devs.extend(st["step_disp_devs"])
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
        else:
            log(f"[chunk {chunk_idx}] fresh start from the root")

        t_chunk = time.time()

        # ---- one chunk: steps until stop / cap / chunk-budget
        while stop is None:
            if step >= STEP_CAP:
                stop = {"kind": "cap", "step": step,
                        "note": "300-step cap reached with neither bar's "
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
            if step in E185_XHASH:
                md5_vs_e185[step] = bool(xh == E185_XHASH[step])
            # ---- the licensed cell's forward/backward/clip (VERBATIM)
            net.zero_grad(set_to_none=True)
            logits, _ = net(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                         1.0))
            assert gnorm > 0.0, "zero gradient — direction undefined"
            # ---- THE ARM'S UPDATE: SGD's direction at Adam's measured size
            g = torch.cat([p.grad.detach().reshape(-1)
                           for p in net.parameters()])
            u = g / torch.norm(g)              # clip-invariant direction
            u_cos_prev = (float(torch.dot(u, prev_u))
                          if prev_u is not None else None)
            i = 0
            with torch.no_grad():
                for p in net.parameters():
                    n = p.numel()
                    p.add_(u[i:i + n].view_as(p), alpha=-STEP_SCALE)
                    i += n
            cur = flat_params(net)
            cum_disp = float(torch.norm(cur - theta0))
            inc_disp = float(torch.norm(cur - prev))
            step_disp_devs.append(abs(inc_disp - STEP_SCALE))
            journal.append({"step": step, "chunk": chunk_idx,
                            "ce_batch": float(loss.item()),
                            "cum_disp": cum_disp, "step_disp": inc_disp,
                            "preclip_gnorm": gnorm, "lr_eff": None,
                            "step_scale": STEP_SCALE, "u_cos_prev": u_cos_prev,
                            "x_md5": xh})
            theta_before = prev.clone()      # theta_{t-1} (densification)
            prev, prev_u = cur, u.clone()

            # ---- the read (EVERY step; forced flags at the gate crossings;
            # one step can cross BOTH gates, so no elif)
            forced_flags = []
            if (not d26_crossed) and cum_disp >= D_SPARE:
                d26_crossed = True
                forced_flags.append("d26")
            if (not d33_crossed) and cum_disp >= D_STOP:
                d33_crossed = True
                forced_flags.append("d33")
            forced = "+".join(forced_flags) if forced_flags else None
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
                prevr = reads[-2]
                s0, s1 = prevr["step"], step
                v0, v1 = prevr["gm12"], gz["mean_pz"]
                d0, d1 = prevr["cum_disp"], cum_disp
                t_x_raw = interp_cross(s0, v0, s1, v1, SHUT_BAR)
                dk_raw = d0 + (t_x_raw - s0) * (d1 - d0)
                stop = {"kind": "kill", "step": step, "t_x_raw": t_x_raw,
                        "bracket_raw": (s0, s1),
                        "gm12_bracket_raw": (v0, v1),
                        "D_kill_interp_raw": dk_raw}
                # ---- along-path densification (refines the bracket ONLY).
                # reads are EVERY step, so s0 = s1-1 and theta_{s0} is the
                # params at the START of this step (theta_before); at a
                # step-1 kill theta_before IS the root (theta0).
                if not SMOKE:
                    th_s0 = theta_before
                    seg = cur - th_s0            # the path segment itself
                    pts = [(0.0, d0, v0)]
                    for f in DENSIFY_F:
                        thf = th_s0 + f * seg
                        df = float(torch.norm(thf - theta0))
                        load_flat(evl, thf)
                        evl.eval()
                        vf = battery_cell(evl, gm12_ids, zid)["mean_pz"]
                        pts.append((f, df, vf))
                        densify_rows.append(
                            {"between": (s0, s1), "f": f, "D": df,
                             "gm12": vf})
                    pts.append((1.0, d1, v1))
                    # tightest adjacent alive->dead pair on the path
                    refined = False
                    for (fa, da, va), (fb, db, vb) in zip(pts, pts[1:]):
                        if va > SHUT_BAR >= vb:
                            f_x = fa + (va - SHUT_BAR) / (va - vb) * (fb - fa)
                            t_x = s0 + f_x * (s1 - s0)
                            d_kill = (da + (f_x - fa) / (fb - fa) * (db - da)
                                      if fb > fa else da)
                            stop.update({
                                "t_x": t_x, "f_x": f_x,
                                "bracket": (s0, s1),
                                "bracket_f": (fa, fb),
                                "gm12_bracket": (va, vb),
                                "D_bracket": (da, db),
                                "D_kill_interp": d_kill})
                            refined = True
                            break
                    if not refined:
                        stop.update({"t_x": t_x_raw,
                                     "bracket": (s0, s1),
                                     "gm12_bracket": (v0, v1),
                                     "D_bracket": (d0, d1),
                                     "D_kill_interp": dk_raw})
                    log(f"  KILL at step {step}: t_x {stop['t_x']:.3f} "
                        f"D_kill(interp) {stop['D_kill_interp']:.4f} "
                        f"bracket D {stop['D_bracket']}")
                break
            if d33_crossed and gz["mean_pz"] > SHUT_BAR:
                stop = {"kind": "spared", "step": step,
                        "gm12_at_d33": gz["mean_pz"],
                        "g0_at_d33": gz0["mean_pz"],
                        "ce_r_at_d33": ce_r, "D_at_d33": cum_disp,
                        "d26_row": (dict(d26_row) if d26_row else None)}
                break

        # ---- chunk end: SAVE the full state (round-trip through disk)
        fp_md5 = flat_md5(net)
        payload = {"model": {k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items()},
                   "gen_state": gen.get_state().clone(),
                   "step": int(step), "chunk": int(chunk_idx),
                   "journal": journal, "reads": reads,
                   "densify_rows": densify_rows,
                   "d26_crossed": bool(d26_crossed),
                   "d33_crossed": bool(d33_crossed),
                   "prev_u": (prev_u.clone() if prev_u is not None
                              else None),
                   "d26_row": d26_row,
                   "md5_vs_e185": dict(md5_vs_e185),
                   "zeph_checks": int(zeph_checks),
                   "step_disp_devs": list(step_disp_devs),
                   "flat_md5": fp_md5,
                   "meta": {"experiment": "opt1c",
                            "arm": "sgrad_dir_at_adam_size",
                            "step_scale": STEP_SCALE,
                            "input_seed": FREEZE_SEED,
                            "base": f"runs/checkpoints/{ROOT_CK}",
                            "parent": "runs/opt1/metrics.json (A0 size) + "
                                      "runs/opt1b/metrics.json (the bleed)"}}
        torch.save(payload, chunk_path)
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
        log(f"[chunk {chunk_idx}] SAVED @ step {step} "
            f"(wall {time.time() - t_chunk:.0f}s, md5 {fp_md5[:10]}…)")
        chunk_idx += 1
        if stop is not None:
            break
        if step >= STEP_CAP:
            stop = {"kind": "cap", "step": step,
                    "note": "300-step cap reached with neither bar's "
                            "trigger inside the window"}
            break

    G_CHUNKS["n_chunks"] = chunk_idx
    G_CHUNKS["pass"] = bool(all(rt["flat_md5_match"]
                                and rt["gen_state_match"]
                                for rt in G_CHUNKS["round_trips"]))
    G_CHUNKS["max_chunk_wall_s"] = max(c["wall_s"] for c in chunks_prov)
    G_DRAWFREE = {"zeph_violations": zeph_checks,
                  "pass": bool(zeph_checks == 0)}
    assert G_DRAWFREE["pass"], "name token leaked into a window"
    G_INPUTS = {"md5_vs_e185_stored": {str(k): v for k, v in
                                       md5_vs_e185.items()},
                "pass": bool(md5_vs_e185
                             and all(md5_vs_e185.values())),
                "note": "input-batch md5s at steps 1..10 vs e185's stored "
                        "hashes (= opt1/opt1b's gate set); all steps' md5s "
                        "recorded in the journal"}
    assert G_INPUTS["pass"], "input stream diverged from e185/opt1/opt1b"
    log(f"G_INPUTS: steps 1..10 md5s match e185's stored hashes: PASS")
    G_STEPSCALE = {
        "step_scale": STEP_SCALE, "committed_ref": STEP_SCALE_COMMITTED,
        "max_abs_step_disp_dev": max(step_disp_devs) if step_disp_devs
        else None,
        "note": "the per-step L2 is PINNED at the measured Adam step size "
                "by construction; this reports the float deviation of the "
                "measured increments",
    }
    log(f"G_STEPSCALE: max |step_disp - STEP_SCALE| "
        f"{G_STEPSCALE['max_abs_step_disp_dev']:.2e} over "
        f"{len(step_disp_devs)} steps")
    log(f"G_CHUNKS: {chunk_idx} chunks, {len(G_CHUNKS['round_trips'])} "
        f"disk round-trips, max chunk wall "
        f"{G_CHUNKS['max_chunk_wall_s']:.0f}s: "
        + ("PASS" if G_CHUNKS["pass"] else "FAIL"))

    # ---- the final checkpoint (provenance for any follow-up cell)
    fin_step = step
    fin_name = ("smoke_" if SMOKE else "") + \
        f"opt1c_sgrad_dir_s{fin_step}"
    torch.save({"model": {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()},
                "meta": {"experiment": "opt1c",
                         "arm": "sgrad_dir_at_adam_size",
                         "steps": int(fin_step),
                         "step_scale": STEP_SCALE,
                         "input_seed": FREEZE_SEED, "stop": stop["kind"],
                         "base": f"runs/checkpoints/{ROOT_CK}",
                         "parent": "runs/opt1/metrics.json + "
                                   "runs/opt1b/metrics.json"}},
               CKPT_DIR / f"{fin_name}.pt")
    log(f"[ckpt] saved {fin_name}.pt (stop={stop['kind']} @ s{fin_step})")

    # =====================================================================
    # ADJUDICATION (registered clauses; composite order frozen:
    # KILLS-AT-GATE -> SPARED-AT-GATE -> CAP-NEITHER; no shopping)
    # =====================================================================
    if stop["kind"] == "spared" and not stop.get("d26_row"):
        stop["d26_row"] = next((r for r in reads
                                if r["cum_disp"] >= D_SPARE), reads[0])
    kill_window_lo, kill_window_hi = KILL_WINDOW
    bars: dict = {}
    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "shakedown only"
    elif stop["kind"] == "kill":
        d_kill = stop["D_kill_interp"]
        in_window = bool(kill_window_lo <= d_kill <= kill_window_hi)
        bars = {
            "KILLS_AT_GATE": {"fires": in_window,
                              "D_kill_interp": d_kill,
                              "window": list(KILL_WINDOW),
                              "in_window": in_window,
                              "t_x": stop["t_x"],
                              "bracket": list(stop["bracket"]),
                              "bracket_f": list(stop["bracket_f"]),
                              "D_bracket": list(stop["D_bracket"]),
                              "gm12_bracket": list(stop["gm12_bracket"]),
                              "raw_bracket": {"t_x": stop["t_x_raw"],
                                              "bracket": list(
                                                  stop["bracket_raw"]),
                                              "D_kill_interp": stop[
                                                  "D_kill_interp_raw"]}},
            "SPARED_AT_GATE": {"fires": False},
            "CAP_NEITHER": {"fires": False},
        }
        if in_window:
            verdict = "KILLS-AT-GATE"
            clause = (f"the fact died at interpolated displacement D_kill "
                      f"{d_kill:.4f} (t_x {stop['t_x']:.2f}; densified "
                      f"bracket D {stop['D_bracket']}, raw bracket D "
                      f"({stop['bracket_raw'][0]},{stop['bracket_raw'][1]}) "
                      f"-> {stop['D_kill_interp_raw']:.4f}) — INSIDE the "
                      f"registered window [{kill_window_lo}, "
                      f"{kill_window_hi}]: cumulative displacement is "
                      f"DIRECTION-ROBUST at Adam's size; the alignment "
                      f"integral is epiphenomenal; 'any path that reaches "
                      f"the gate kills' holds in its letter — at Adam's "
                      f"measured size, even the raw-gradient direction "
                      f"(SGD's own) kills at the gate.")
        else:
            verdict = "KILL-OUT-OF-WINDOW (no bar)"
            clause = (f"the fact died at interpolated displacement D_kill "
                      f"{d_kill:.4f} (t_x {stop['t_x']:.2f}) — OUTSIDE the "
                      f"registered window [{kill_window_lo}, "
                      f"{kill_window_hi}]: NEITHER bar fires (opt1b's "
                      f"convention: a kill outside the window is a graded "
                      f"outcome; the full fact-vs-D curve is reported "
                      f"verbatim; no bar claim).")
    elif stop["kind"] == "spared":
        d26 = stop.get("d26_row")
        bars = {
            "KILLS_AT_GATE": {"fires": False},
            "SPARED_AT_GATE": {"fires": True,
                               "d26_row": d26,
                               "gm12_at_d33": stop["gm12_at_d33"],
                               "D_at_d33": stop["D_at_d33"],
                               "step_at_d33": stop["step"]},
            "CAP_NEITHER": {"fires": False},
        }
        verdict = "SPARED-AT-GATE"
        clause = (f"the trajectory PASSED D = 2.6 with the fact ALIVE "
                  f"(g-12 {d26['gm12']:.4f} > {SHUT_BAR} at D "
                  f"{d26['cum_disp']:.4f}, step {d26['step']}) and stayed "
                  f"alive through D = {D_STOP} (g-12 "
                  f"{stop['gm12_at_d33']:.4f} at D "
                  f"{stop['D_at_d33']:.4f}, step {stop['step']}; g0 "
                  f"{stop['g0_at_d33']:.4f}, CE_R "
                  f"{stop['ce_r_at_d33']:.4f}) — the kill is NOT carried "
                  f"by size alone; DIRECTION (or normalization structure) "
                  f"carries it; 'any path reaching the gate kills' dies in "
                  f"its current letter and the law moves to ruler-aligned-"
                  f"displacement currency. TENSION TO REPORT HONESTLY: "
                  f"e188's RAW-WINS read death as priced in raw "
                  f"displacement across the lr grid under AdamW — this "
                  f"arm, at Adam's own size but SGD's direction, survives "
                  f"the raw gate; the two results must be reconciled "
                  f"(e188's grid never varied direction at fixed size).")
    else:
        fin = journal[-1]
        tail = [r["step_disp"] for r in journal[-40:]]
        rate = sum(tail) / max(len(tail), 1)
        bars = {
            "KILLS_AT_GATE": {"fires": False},
            "SPARED_AT_GATE": {"fires": False},
            "CAP_NEITHER": {"fires": True},
            "projection": {"steady_disp_rate": rate,
                           "label": "EXTRAPOLATED — projections never "
                                    "adjudicate"},
        }
        verdict = "CAP-NEITHER"
        clause = (f"neither fired within the {STEP_CAP}-step cap (final "
                  f"|d| {fin['cum_disp']:.4f}, last g-12 "
                  f"{reads[-1]['gm12']:.4f}); the curve is reported; no "
                  f"projection adjudicates.")
    log("=" * 78)
    log(f"OPT1C VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # ---- the arm's fact-vs-D curve + alignment summary (co-reads)
    curve = [{"step": r["step"], "D": r["cum_disp"], "gm12": r["gm12"],
              "g0": r["g0"], "ce_r": r["ce_r"],
              "cos_delta_fact_g0": r["cos_delta_fact_g0"],
              "cos_delta_fact_m12": r["cos_delta_fact_m12"],
              "provenance": "opt1c (this run)"}
             for r in reads]
    my_reads = [r for r in reads if r["step"] > 0]
    align_summary = None
    if my_reads:
        early = [r["cos_delta_fact_g0"] for r in my_reads
                 if r["cum_disp"] <= 1.0]
        late = [r["cos_delta_fact_g0"] for r in my_reads
                if r["cum_disp"] > 1.0]
        align_summary = {
            "mean_cos_g0_D_le_1": (sum(early) / len(early)) if early
            else None,
            "mean_cos_g0_D_gt_1": (sum(late) / len(late)) if late else None,
            "min_cos_g0": min(r["cos_delta_fact_g0"] for r in my_reads),
            "max_cos_g0": max(r["cos_delta_fact_g0"] for r in my_reads),
            "note": "cumulative-displacement alignment, opt1's convention; "
                    "negative = death-aligned; opt1's arms sat at "
                    "-0.015..-0.105, opt1b's bleed at -0.025..-0.041",
        }

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "opt1c_direction_size",
        "date": common.now_iso(),
        "registration": ("the dispatch's registration IS the registration "
                         "(the three bars quoted verbatim in the module "
                         "docstring and in registered_prediction, frozen "
                         "before compute)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("inside the Adam kill, is it the raw-gradient "
                     "DIRECTION at small size that spares (SIZE carries "
                     "the kill), or would ANY direction at Adam's step "
                     "size kill (cumulative displacement is direction-"
                     "robust)? ONE arm splits direction from size: the "
                     "raw batch-gradient DIRECTION (SGD's own, "
                     "g_t/||g_t||_2, descent) at Adam's measured per-step "
                     "L2 on the licensed e185 wash cell — 'SGD's "
                     "direction at Adam's size'"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "e185 arm C (CONTROL) VERBATIM — the "
                             "consolidated host fact under its neutral/"
                             "corpus wash; opt1's machinery verbatim",
            "batch": f"32 = {ANCH_BS} neutral-bank anchors (seed "
                     f"{E170_ANCHOR_SEED}, fixed) + {RAND_BS} random "
                     "corpus windows, full-token CE, clip 1.0",
            "targets": "TRUE (the real neutral wash — the update rule is "
                       "the only delta vs the licensed cell)",
            "input_seed": FREEZE_SEED,
            "update_rule": f"delta_t = {STEP_SCALE!r} * (g_t/||g_t||_2), "
                           "theta -= delta_t (DESCENT along the raw "
                           "gradient direction; g_t post-clip — direction "
                           "clip-invariant, pre-clip norm recorded; NO "
                           "optimizer, NO lr)",
            "measure_light": "g-12 / g0 batteries + CE_R + alignment + "
                             "displacement + pre-clip grad norms + "
                             "direction persistence (the dispatch's read "
                             "list; no census/deletions)",
        },
        "provenance": {
            "step_scale_measured": STEP_SCALE,
            "step_scale_committed_opt1_a0": STEP_SCALE_COMMITTED,
            "t0_gate": G_T0,
            "hashes_vs_opt1_committed_t0": {
                "step1_x_md5": x1_md5,
                "step1_x_md5_matches_e185_stored":
                    bool(x1_md5 == E185_XHASH[1]),
                "step1_ce_batch_measured": ce1,
                "step1_ce_batch_opt1_committed": CE1_COMMITTED,
                "step1_preclip_gnorm_measured": gn1,
                "step1_preclip_gnorm_opt1_committed": GN1_COMMITTED,
                "adamw_one_step_L2_measured": disp1,
                "adamw_one_step_L2_opt1_committed": STEP_SCALE_COMMITTED},
            "effective_step1_sgd_lr": STEP_SCALE / gn1,
            "resume": {
                "method": ("fresh start from the root (no replay needed — "
                           "the arm is a NEW trajectory, not a "
                           "continuation); chunk state round-trips through "
                           "runs/opt1c/chunk_state.pt per the opt1b "
                           "convention; the stream position carries in the "
                           "generator state"),
                "state_hash_chain": [c["flat_md5_at_end"]
                                     for c in chunks_prov]},
        },
        "arm": {
            "tag": "sgrad_dir_at_adam_size",
            "parent_size": "runs/opt1/metrics.json arms.a0_adamw_ref "
                           "(step-1 L2 — the size this arm pins)",
            "parent_bleed": "runs/opt1b/metrics.json (the raw-gradient "
                            "path at its own small size — the contrast "
                            "arm, committed)",
            "max_steps": STEP_CAP, "steps_ran": fin_step,
            "train_seconds": round(sum(c["wall_s"] for c in chunks_prov),
                                   1),
            "time_cap_per_chunk": CHUNK_CAP,
            "seed": FREEZE_SEED,
            "traj": journal, "ckpt_table": reads,
            "densify": densify_rows,
            "zeph_violations": zeph_checks,
        },
        "chunks": chunks_prov,
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT, "G_T0": G_T0, "G_INPUTS": G_INPUTS,
                  "G_DRAWFREE": G_DRAWFREE, "G_CHUNKS": G_CHUNKS,
                  "G_STEPSCALE": G_STEPSCALE},
        "references": {
            "opt1": {"metrics": "runs/opt1/metrics.json",
                     "role": "the parent cell: A0's measured step-1 L2 "
                             "(the pinned size), the t=0 bit-identity "
                             "references, and the Adam arms' fact-vs-D "
                             "curves (the overlay, plotted from committed "
                             "data, never rerun)"},
            "opt1b": {"metrics": "runs/opt1b/metrics.json",
                      "role": "the bleed (the raw-gradient path at its own "
                              "small size): the fact-vs-D curve overlaid "
                              "from committed data; T142's CAP-NEITHER"},
            "e185": {"metrics": "runs/e185/metrics.json",
                     "role": "the licensed cell + the stored input md5s "
                             "this run re-derives at steps 1..10"},
        },
        "fact_vs_D_curve": curve,
        "adam_overlay": {t: [{"step": r["step"], "D": r["cum_disp"],
                              "gm12": r["gm12"]}
                             for r in rows if r["step"] > 0]
                         for t, rows in adam_arms.items()},
        "bleed_overlay": [{"step": r["step"], "D": r["D"], "gm12": r["gm12"]}
                          for r in bleed_curve],
        "coreads": {
            "alignment": align_summary,
            "ce_r": {"note": "organism health at every checkpoint (rows "
                             "in fact_vs_D_curve / arm.ckpt_table)",
                     "root": root_cells["ce_r"],
                     "final": reads[-1]["ce_r"] if reads else None},
            "direction_persistence": {
                "u_cos_prev_rows": [r["u_cos_prev"] for r in journal
                                    if r["u_cos_prev"] is not None],
                "note": "cos(u_t, u_{t-1}) per step — why cumulative D "
                        "grows sublinearly (random |cos| ~ 6e-4 in 2.7M "
                        "dims; consecutive stream gradients persist)"},
        },
        "adjudication": {
            "bars": bars,
            "verdict": verdict, "clause": clause,
            "stop": {k: v for k, v in stop.items()},
            "constants": {"step_scale": STEP_SCALE,
                          "D_spare_gate": D_SPARE, "D_stop": D_STOP,
                          "kill_window": list(KILL_WINDOW),
                          "adam_gate_band": list(ADAM_GATE_BAND),
                          "shut_bar": SHUT_BAR, "step_cap": STEP_CAP,
                          "D_kill_adam_A0": D_KILL_ADAM},
            "composite_order": "KILLS-AT-GATE -> SPARED-AT-GATE -> "
                               "CAP-NEITHER (frozen before compute; kill "
                               "wins tie steps)",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1, single seed lineage (10902; the "
                            "replicate ladder is a separate dispatch "
                            "decision); ONE root, ONE input stream "
                            "(bit-gated vs e185/opt1 at steps 1..10 + "
                            "the t=0 gate); single family (the e131 "
                            "consolidated line); this arm differs from "
                            "the licensed cell in ONE thing only — the "
                            "update rule (direction x size factorial's "
                            "single cell)"),
            "extrapolation_free": ("every adjudication input is MEASURED "
                                   "inside this run's window: the kill "
                                   "(if any) is bracketed by every-step "
                                   "reads and refined by along-path "
                                   "densification (points ON the same "
                                   "trajectory); the spare is certified "
                                   "by forced reads at the measured 2.6- "
                                   "and 3.3-crossing steps; CAP-NEITHER "
                                   "carries no adjudicating projection "
                                   "by registration"),
            "float_texture": ("CPU fp32, this process, 4 threads (the "
                              "e185/opt1 reduction order); the t=0 gate "
                              "reproduces opt1's committed A0 row on "
                              "THIS device before any training (Rule 12); "
                              "the parents' curves are loaded from "
                              "committed metrics, never rerun"),
            "committed_data_reuse": ("opt1's Adam arms and opt1b's bleed "
                                     "are plotted from runs/opt1/"
                                     "metrics.json + runs/opt1b/"
                                     "metrics.json verbatim (the "
                                     "dispatch's order; no parent reruns)"),
            "no_guarantees": ("nothing was guaranteed ex ante: the arm "
                              "could kill at the gate, pass it alive, or "
                              "bleed like SGD — the openness is the "
                              "point; the stop actually observed is "
                              f"'{stop['kind']}'"),
            "interpolation_texture": ("D_kill/t_x are linear "
                                      "interpolations inside measured "
                                      "brackets (raw = one step wide; "
                                      "densified = a 0.2-fraction sub-"
                                      "bracket ON the path); the bracket "
                                      "is the honest resolution limit — "
                                      "both are reported"),
            "one_arm": ("the factorial's other cell (Adam's direction at "
                        "SGD's size — the mirror arm) was NOT run: the "
                        "dispatch registered one arm, minimum; if this "
                        "arm kills at the gate the mirror is moot for "
                        "the registered question, if it survives the "
                        "mirror becomes the next cell"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": {
            fin_name: {"path": f"runs/checkpoints/{fin_name}.pt",
                       "arm": "sgrad_dir_at_adam_size",
                       "steps": int(fin_step), "stop": stop["kind"]},
            "chunk_state": {"path": "runs/opt1c/chunk_state.pt",
                            "note": "the resumable state (model + "
                                    "generator + step + journal + reads)"},
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

    plot_fact_vs_D(rd / "opt1c_fact_vs_D.png", curve, adam_arms,
                   bleed_curve, verdict, clause, stop)
    plot_coreads(rd / "opt1c_coreads.png", curve, journal, reads,
                 chunks_prov, stop, STEP_SCALE)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'opt1c_fact_vs_D.png'}, "
        f"{rd / 'opt1c_coreads.png'}, journal.jsonl, chunk_state.pt, "
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


def plot_fact_vs_D(path, curve, adam_arms, bleed_curve, verdict, clause,
                   stop):
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    # the bleed (opt1b committed)
    ax.plot([r["D"] for r in bleed_curve], [r["gm12"] for r in bleed_curve],
            "o-", ms=4.5, lw=1.8, color="royalblue", alpha=0.9,
            label="SGD 1e-2 — THE BLEED (opt1b committed, to s600)")
    # the Adam arms (opt1 committed)
    for t, rows in adam_arms.items():
        col, lbl = ADAM_STYLE[t]
        pts = [(r["D"], r["gm12"]) for r in rows if r["step"] > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--", ms=5,
                lw=1.5, color=col, alpha=0.85, label=lbl)
    # THIS arm
    ax.plot([r["D"] for r in curve], [r["gm12"] for r in curve], "D-",
            ms=7, lw=2.6, color="magenta", alpha=0.95, zorder=5,
            label="SGD DIRECTION at Adam SIZE 1.6543/step (opt1c, this run)")
    ax.axvspan(KILL_WINDOW[0], KILL_WINDOW[1], color="tab:purple",
               alpha=0.08, label="kill window [2.12, 3.27] (2.49-2.84 ±15%)")
    ax.axvline(D_SPARE, ls=":", lw=2.0, color="navy", alpha=0.8,
               label=f"D=2.6 spared gate")
    ax.axvline(D_STOP, ls=":", lw=1.4, color="magenta", alpha=0.6,
               label=f"D=3.3 alive-stop (window top + margin)")
    for yv, col, lbl in ((0.50, "seagreen", "0.50 SPARE"),
                         (SHUT_BAR, "tab:purple", f"{SHUT_BAR} DISSOLVE")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    if stop["kind"] == "kill":
        ax.plot([stop["D_kill_interp"]], [SHUT_BAR], "*", ms=19,
                color="yellow", mec="k", zorder=6,
                label=f"KILL (interp D {stop['D_kill_interp']:.3f})")
    if stop["kind"] == "spared":
        ax.plot([stop["d26_row"]["cum_disp"]], [stop["d26_row"]["gm12"]],
                "*", ms=19, color="lime", mec="k", zorder=6,
                label=f"passed D=2.6 ALIVE (g-12 "
                      f"{stop['d26_row']['gm12']:.3f})")
        ax.plot([stop["D_at_d33"]], [stop["gm12_at_d33"]], "*", ms=15,
                color="lime", mec="k", alpha=0.6, zorder=6,
                label=f"alive at D=3.3 (g-12 {stop['gm12_at_d33']:.3f})")
    ax.set_xlabel(r"cumulative displacement $\|\theta_t-\theta_0\|_2$ "
                  "(2.74M params)")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.08, 5.3)
    ax.legend(fontsize=7.0, loc="lower left")
    ax.set_title("THE DIRECTION-SIZE FACTORIAL — SGD's direction at Adam's "
                 "measured size (three-way overlay, committed parents)",
                 fontsize=10)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"OPT1C VERDICT: {verdict}", fontsize=10.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.05
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=7.3, va="top",
                family="monospace")
        y -= 0.026
    y -= 0.02
    ax.text(0.02, y, "  step     D        g-12    g0     CE_R    cos(g0)  "
                     " cos(m12)", fontsize=7.2, va="top", family="monospace")
    y -= 0.028
    for c in curve:
        ax.text(0.02, y,
                f"  {c['step']:>5d}  {c['D']:7.4f}  {c['gm12']:.4f}  "
                f"{c['g0']:.4f}  {c['ce_r']:.4f}  "
                + (f"{c['cos_delta_fact_g0']:+.4f}" if
                   c["cos_delta_fact_g0"] is not None else "    n/a")
                + "  "
                + (f"{c['cos_delta_fact_m12']:+.4f}" if
                   c["cos_delta_fact_m12"] is not None else "    n/a"),
                fontsize=6.9, va="top", family="monospace",
                color="magenta")
        y -= 0.026
    fig.suptitle("OPT1C — THE DIRECTION-SIZE FACTORIAL: does ANY direction "
                 "at Adam's size kill, or does the raw-gradient direction "
                 "spare?", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coreads(path, curve, journal, reads, chunks_prov, stop,
                 step_scale):
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    # (0,0) D(t) vs step + direction persistence
    ax = axes[0, 0]
    ax.plot([r["step"] for r in journal], [r["cum_disp"] for r in journal],
            "-", lw=1.8, color="magenta",
            label=r"cumulative $\|\theta_t-\theta_0\|_2$")
    ax.axhline(D_SPARE, ls=":", lw=1.8, color="navy", alpha=0.9,
               label="D=2.6 spared gate")
    ax.axhspan(ADAM_GATE_BAND[0], ADAM_GATE_BAND[1], color="tab:purple",
               alpha=0.10, label="Adam kill band 2.49-2.84 (opt1)")
    ax.axhline(D_KILL_ADAM, ls="--", lw=1.0, color="crimson", alpha=0.7,
               label="A0 D_kill 2.489")
    for c in chunks_prov[:-1]:
        ax.axvline(c["step_to"], ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"$\|\theta_t-\theta_0\|_2$")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("displacement vs step (each step pinned at "
                 f"{step_scale:.4f} L2; chunk boundaries dotted)",
                 fontsize=10)

    # (0,1) the size panel: per-step |d| vs pre-clip ||g||
    ax = axes[0, 1]
    ax.scatter([r["preclip_gnorm"] for r in journal],
               [r["step_disp"] for r in journal], s=22, color="magenta",
               alpha=0.8, label="this arm (direction normalized)")
    ax.axhline(step_scale, ls="--", lw=1.6, color="k", alpha=0.8,
               label=f"STEP_SCALE {step_scale:.4f} (Adam's measured size)")
    ax.axhline(1e-3 * (2739072 ** 0.5), ls=":", lw=1.2, color="crimson",
               alpha=0.7, label=r"lr$\sqrt{N}$ = 1.655 (Adam's arithmetic)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"pre-clip $\|g\|_2$ (per step)")
    ax.set_ylabel(r"per-step $\|\Delta\theta\|_2$")
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE SIZE SPLIT — this arm moves Adam's size along SGD's "
                 "direction regardless of ||g||", fontsize=9.5)

    # (1,0) alignment vs D
    ax = axes[1, 0]
    rows = [c for c in curve if c["cos_delta_fact_g0"] is not None]
    ax.plot([c["D"] for c in rows], [c["cos_delta_fact_g0"] for c in rows],
            "o-", ms=5, lw=1.8, color="teal", label=r"cos($\Delta\theta$, "
                                                    r"$\nabla$g0)")
    ax.plot([c["D"] for c in rows], [c["cos_delta_fact_m12"] for c in rows],
            "s--", ms=4.5, lw=1.5, color="olive", alpha=0.8,
            label=r"cos($\Delta\theta$, $\nabla$m12)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("cumulative displacement D")
    ax.set_ylabel("alignment cos")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("THE ALIGNMENT CO-READ — negative = death-aligned "
                 "(critic's convention)", fontsize=9.5)

    # (1,1) CE_R vs D + direction persistence inset
    ax = axes[1, 1]
    ax.plot([c["D"] for c in curve], [c["ce_r"] for c in curve], "o-",
            ms=5, lw=1.8, color="saddlebrown", label="CE_R (organism)")
    ax.axhline(curve[0]["ce_r"], ls="--", lw=1.0, color="gray",
               alpha=0.8, label=f"root CE_R {curve[0]['ce_r']:.3f}")
    ax.set_xlabel("cumulative displacement D")
    ax.set_ylabel("CE_R")
    ax.legend(fontsize=7.5, loc="upper left")
    urows = [r for r in journal if r["u_cos_prev"] is not None]
    if urows:
        ax2 = ax.twinx()
        ax2.plot([r["step"] for r in urows],
                 [r["u_cos_prev"] for r in urows], "v:", ms=5, lw=1.2,
                 color="slategray", alpha=0.8)
        ax2.set_ylabel("cos(u_t, u_{t-1})  (dotted, right)",
                       color="slategray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="slategray", labelsize=7)
    ax.set_title("organism health + per-step direction persistence",
                 fontsize=10)

    fig.suptitle("OPT1C — co-reads: displacement, the size split, "
                 "alignment, the organism", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
