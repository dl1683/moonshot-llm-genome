"""OPT1B — THE DIRECT SGD KILL (opt1's owed discriminator; T139's cell).

WHY: opt1 (runs/opt1/metrics.json, T139) decomposed the two-step wash
kill into CLOCK (Adam's sign-normalization: 1687x/step at matched lr)
and GATE (displacement ~2.49-2.84, clock-invariant across Adam
variants). But every SGD arm was CPU-cap-limited (69-97 steps), so the
SGD kill clock was located only by extrapolation (~379 steps at lr
1e-2) and NEVER adjudicated. Under matched-lr SGD the same corpus wash
STRENGTHENED the fact (g-12 0.916 -> 0.940-0.955 at |d| <= 0.29) — the
stream that kills in two steps under Adam TEACHES under SGD. opt1b
measures the SGD trajectory's fate DIRECTLY: continue opt1's A2b arm
(SGD lr 1e-2) to EITHER the kill (g-12 <= 0.27, e185's kill def,
interpolated) OR D = 2.6 (15% past the Adam gate) reached alive,
whichever fires first; cap total at ~600 steps.

REGISTERED BARS (frozen here before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - SGD-KILLS-AT-GATE: "fires if the fact dies at displacement D_kill
    within 2.49-2.84 ± 15% (i.e. [2.12, 3.27]) — the displacement gate
    GENERALIZES across learned trajectory classes; the trajectory
    hypothesis (T137) keeps its two-class form (learned paths and
    static jumps)."
  - SGD-SPARED-AT-GATE: "fires if the trajectory passes D = 2.6 with
    the fact alive (g-12 > 0.27) — the gate is TRAJECTORY-CLASS-TYPED
    (sign-normalized paths kill, raw-gradient paths are spared at the
    same displacement); the law gains a third clause and the paper's
    optimizer clause strengthens."
  - CAP-NEITHER: "neither fires within 600 steps — report the
    fact-vs-D curve and the projected crossing, never adjudicated."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * the arm = opt1's A2b VERBATIM: plain SGD (momentum 0, wd 0, clip
    1.0 retained) at constant lr 1e-2 on the licensed e185 wash cell
    (root e131_consolidated_e113, batch 32 = 16 neutral-bank anchors
    seed 170 + 16 random corpus windows, TRUE targets, seed-10902
    stream, full-token CE) — lab/opt1_optimizer_controls.py's
    machinery reused VERBATIM (copied, not imported, to own the device
    policy; e185's own provenance carries down).
  * the fact = g-12 (install-60 battery mean p(Z) at ctx offset -12,
    e185's convention); KILL = g-12 <= 0.27 (the arc's SHUT bar),
    interpolated: t_x = the linear-in-step crossing between the last
    alive read and the first dead read; D_kill = the same-bracket
    linear interpolation of cumulative displacement at t_x.
  * displacement = cumulative ||theta_t - theta_0||_2 over all
    2,739,072 params (fp32, CPU, measured every step), e185/opt1's
    currency verbatim.
  * D = 2.6 = the SPARED gate (15% past the Adam gate; the dispatch's
    frozen number); "reached alive" = g-12 > 0.27 at a FORCED read at
    the first step whose cumulative displacement crosses 2.6 (the
    crossing step is known exactly — displacement is measured every
    step).
  * kill-vs-spared precedence = "whichever first": the kill wins iff
    its interpolated t_x precedes the measured D=2.6 crossing step;
    since the crossing forces a read, an alive crossing CERTIFIES
    SGD-SPARED-AT-GATE, and any dead read before it CERTIFIES the
    kill.
  * cap = 600 steps total from the root (the dispatch's ~600); if
    neither fires, CAP-NEITHER reports the curve + the PROJECTED
    crossing (labeled EXTRAPOLATED; projections never adjudicate).

RESUME / STREAM-CONTINUITY METHOD (documented exactly, per the
dispatch): runs/checkpoints/opt1_a2b_* does NOT exist (opt1 saves
non-A0 arms only at final steps that land on its checkpoint ladder;
a2b's time-cap step 69 is not on it) — verified by scan at run time.
The chunk state is therefore re-established BY DETERMINISTIC REPLAY
from the root under the same seed conventions: plain SGD(momentum 0)
is STATELESS (no optimizer state to carry), and the wash stream is
fully determined by torch.Generator().manual_seed(10902) with exactly
two randint draws per step (aj then rj — opt1's opt_wash draw order).
The replay covers steps 1..69 and is GATED per step against opt1's
committed a2b trajectory (ce_batch, cum_disp, preclip_gnorm; opt1's
5e-6 bit / 0.05 fallback tolerance convention) plus the input-batch
md5s at steps 1..10 vs e185's stored hashes — this IS the resumed
state's identity check (Rule 12: displacement continuity within float
tolerance). Continuation then runs in ckpt-RESUMABLE CHUNKS of <= 180
s: after each chunk the full state (model state_dict + generator
state + step counter + journal) round-trips through disk
(runs/opt1b/chunk_state.pt) and the next chunk LOADS it (the load path
a fresh process would take); the stream position carries in the
generator state, so the input stream CONTINUES bit-identically, never
repeats (per-step md5s recorded; battery reads consume no RNG).
Chunk-boundary parameter md5s form a hash chain (the identity a
cross-process resumer must match).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any continuation):
battery geometry, splice mix, corpus ZEPH count 0, neutral bank
bit-equal to e185's stored starts, root gated vs e151's before-cells
(opt1's gate set verbatim) + the replay-identity gate above. WHAT THE
ARM GUARANTEES: NOTHING. The SGD path could kill at the gate, pass it
alive, or creep forever — that openness is the point.

CO-READS: (1) the fact-vs-displacement curve with opt1's Adam arms
OVERLAID from the committed runs/opt1/metrics.json (plotted from
committed data, never rerun); (2) alignment cos(delta_theta_t, grad
g0_t) at every checkpoint (the W022/W023 question: does SGD's path
drift more or less death-directed as D grows? critic's sign
convention: negative = death-aligned; co-reported for the m12 ruler);
(3) CE_R at every checkpoint (does the organism keep learning?).

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is another agent's — never claimed), torch
threads 4 (the e185/opt1 reduction order), SEQUENTIAL chunks <= 180 s
each (chunk count documented; in-chunk checkpoint reads count toward
the cap, opt1's FT_TIME_CAP convention), n=1, single seed lineage
(A2b's = 10902), no reruns beyond the 600-step cap.

Outputs: runs/opt1b/{metrics.json, opt1b_fact_vs_D.png,
opt1b_coreads.png, chunk_state.pt, journal.jsonl}; checkpoint
runs/checkpoints/opt1b_a2b_sgd_1e-2_s<final>.pt. No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python opt1b_sgd_kill.py    (OPT1B_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import glob as _glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185/e187/opt1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185/opt1-era reduction order

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("OPT1B_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "opt1b is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (opt1's, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E185_METRICS = E43.REPO / "runs" / "e185" / "metrics.json"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the A2b arm + the continuation envelope (dispatch-frozen) ------------------
LR_A2B = 1e-2                     # opt1's A2b lr VERBATIM
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185/opt1)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
CHUNK_CAP = 178.0                 # dispatch: <=180 s per chunk (margin for step overshoot)
STEP_CAP = 600                    # dispatch: ~600-step total cap
D_SPARE = 2.6                     # the SPARED gate (15% past the Adam gate; frozen)
KILL_WINDOW = (2.12, 3.27)        # 2.49-2.84 +- 15% (the registered letter's window)
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE)
SURVIVE_NOTE = 0.50               # SPARE reference line (opt1's convention; report-only here)
E170_ANCHOR_SEED = 170            # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
ADAM_GATE_BAND = (2.49, 2.84)     # opt1's measured Adam kill displacements (T139)

# ---- checkpoint cadence for the CONTINUATION (replay reuses opt1's stored rows) -
FIRST_READ = 80                   # opt1's stored a2b reads cover {0,1,2,4,8,10,20,40}
GAP_HI, GAP_MID, GAP_LO = 40, 10, 4   # densify as g-12 falls (0.55 / 0.35 triggers)

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
    "sgd_kills_at_gate": "SGD-KILLS-AT-GATE: \"fires if the fact dies "
        "at displacement D_kill within 2.49-2.84 ± 15% (i.e. [2.12, "
        "3.27]) — the displacement gate GENERALIZES across learned "
        "trajectory classes; the trajectory hypothesis (T137) keeps "
        "its two-class form (learned paths and static jumps).\"",
    "sgd_spared_at_gate": "SGD-SPARED-AT-GATE: \"fires if the "
        "trajectory passes D = 2.6 with the fact alive (g-12 > 0.27) — "
        "the gate is TRAJECTORY-CLASS-TYPED (sign-normalized paths "
        "kill, raw-gradient paths are spared at the same displacement); "
        "the law gains a third clause and the paper's optimizer clause "
        "strengthens.\"",
    "cap_neither": "CAP-NEITHER: \"neither fires within 600 steps — "
        "report the fact-vs-D curve and the projected crossing, never "
        "adjudicated.\"",
    "operationalizations": "the arm = opt1's A2b VERBATIM (plain SGD "
        "m0/wd0/clip1.0, constant lr 1e-2, seed-10902 licensed stream); "
        "KILL = g-12 <= 0.27 interpolated linearly-in-step between the "
        "last alive and first dead read; D_kill = same-bracket linear "
        "interpolation of cumulative displacement; D = 2.6 'reached "
        "alive' = a FORCED read at the first step whose cumulative "
        "displacement crosses 2.6 shows g-12 > 0.27; precedence = "
        "whichever first (the kill wins iff its t_x precedes the "
        "measured 2.6-crossing step); cap = 600 steps total from the "
        "root; CAP-NEITHER's projection is labeled EXTRAPOLATED and "
        "never adjudicates; composite order frozen SGD-KILLS-AT-GATE -> "
        "SGD-SPARED-AT-GATE -> CAP-NEITHER; a kill OUTSIDE the window "
        "[2.12, 3.27] fires NO bar (graded outcome, curve reported "
        "verbatim).",
    "registration": "the dispatch's registration IS the registration "
        "(the three bars quoted verbatim in the module docstring and "
        "here, frozen before compute). Adjudicate against exactly this; "
        "no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "Checkpoint cadence: opt1's stored a2b reads {0,1,2,4,8,10,20,40} "
    "are REUSED from runs/opt1/metrics.json (committed data, same "
    "stream — never rerun); the continuation reads at step 80 then "
    "adaptive gaps (40 -> 10 -> 4 as g-12 falls through 0.55/0.35) + a "
    "FORCED read at the D=2.6 crossing + a final read at the cap/stop. "
    "The replay itself (steps 1..69) performs NO battery reads (the "
    "per-step ce/displacement/grad-norm continuity gate is read-free) "
    "— this is why chunk timings differ from opt1's wall clock at the "
    "same steps, while the STREAM and ARITHMETIC stay bit-identical.",
    "Resume method (the dispatch's documented fallback): no "
    "opt1_a2b_* checkpoint exists on disk (verified by scan; opt1 "
    "saved non-A0 arms only at ladder-final steps), so the A2b state "
    "is re-established by DETERMINISTIC REPLAY from the root (SGD "
    "m=0 is stateless; the stream is fixed by manual_seed(10902) + "
    "two randint draws/step in opt1's order aj-then-rj). The replay "
    "is gated per step vs opt1's committed a2b trajectory + input "
    "md5s at steps 1..10 vs e185's stored hashes. Continuation chunks "
    "round-trip the full state (model + generator state + step + "
    "journal) through runs/opt1b/chunk_state.pt; each chunk LOADS the "
    "previous chunk's file (the cross-process path), and chunk-boundary "
    "parameter md5s form a hash chain.",
    "Chunk cap 178 s (dispatch <=180 s; the margin absorbs one ~2 s "
    "step overshoot so every chunk's wall stays <= 180 s); in-chunk "
    "checkpoint reads count toward the cap (opt1's FT_TIME_CAP "
    "convention). Total cap 600 steps from the root (= the dispatch's "
    "~600), of which 69 are the replay.",
    "Light reads only (g-12/g0 batteries + CE_R + alignment + "
    "displacement + pre-clip grad norms) — the dispatch's read list; "
    "no census/deletions. CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before "
    "torch; the GPU is another agent's, never claimed); threads 4; "
    "n=1; single seed lineage (10902, A2b's); no reruns beyond the cap.",
    "Smoke mode trims: replay gate to the first 4 steps, cap 8 steps, "
    "12 s chunks, first read at step 5 gap 2; nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/opt1_optimizer_controls.py VERBATIM (whose own provenance is
# lab/e185_noise_wash.py via e187's verified copies — the e176n lineage).
# Copied rather than imported to own the device policy.

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
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
    rd = run_dir("opt1b_smoke" if SMOKE else "opt1b")
    chunk_path = rd / "chunk_state.pt"
    log(f"OPT1B THE DIRECT SGD KILL (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), sequential "
        f"chunks <= {CHUNK_CAP + 2:.0f}s, cap {STEP_CAP} steps, n=1, seed "
        f"lineage {FREEZE_SEED} (A2b's)")

    # ---------------- opt1 references (committed data; the parent cell)
    if not OPT1_METRICS.exists():
        raise RuntimeError(f"missing parent metrics: {OPT1_METRICS}")
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    a2b = opt1m["arms"]["a2b_sgd_1e-2"]
    REPLAY_TO = int(a2b["steps_ran"])
    stored_traj = {int(r["step"]): r for r in a2b["traj"]}
    stored_ckpt = a2b["ckpt_table"]
    adam_arms = {t: opt1m["arms"][t]["ckpt_table"]
                 for t in ("a0_adamw_ref", "a3_adamw_warmup30",
                           "a4_adamw_b2_0.999", "a5_adamw_moment_reset")}
    a2b_tstar_opt1 = a2b["tstar"]
    a2b_ckpts_on_disk = sorted(
        p.replace("\\", "/").split("/")[-1]
        for p in _glob.glob(str(CKPT_DIR / "opt1_a2b*")))
    if SMOKE:
        REPLAY_TO = min(REPLAY_TO, 4)
    log(f"parent: opt1 a2b steps_ran {a2b['steps_ran']} final|d| "
        f"{a2b_tstar_opt1['final_cum_disp']:.4f} rate "
        f"{a2b_tstar_opt1['steady_disp_rate']:.5f}/step proj "
        f"~{a2b_tstar_opt1['extrapolated_steps_to_D_kill']:.0f} steps "
        f"(EXTRAPOLATED); opt1 verdict {opt1m['adjudication']['verdict']}")
    log(f"a2b checkpoints on disk: {a2b_ckpts_on_disk or 'NONE'}")

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

    # =====================================================================
    # THE CHUNKED CONTINUATION (replay 1..REPLAY_TO, then to the stop)
    # =====================================================================
    n_anc = anchor_neutral.shape[0]
    journal: list[dict] = []          # per-step rows (persisted per chunk)
    reads: list[dict] = []            # checkpoint battery rows (persisted)
    chunks_prov: list[dict] = []      # chunk bookkeeping
    replay_diffs: dict[str, list[float]] = {"ce": [], "disp": [], "gn": []}
    md5_vs_e185: dict[int, bool] = {}
    zeph_checks = 0
    stop: dict | None = None
    d26_crossed = False
    G_CHUNKS = {"round_trips": [], "pass": None}

    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=LR_A2B)   # plain: m=0, wd=0
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    evl = copy.deepcopy(net0)
    evl.eval()
    prev = theta0.clone()
    step = 0
    chunk_idx = 0

    def read_gap() -> int:
        if not reads:
            return max(1, FIRST_READ - REPLAY_TO)
        last = reads[-1]["gm12"]
        if last is None or last >= 0.55:
            return GAP_HI
        if last >= 0.35:
            return GAP_MID
        return GAP_LO

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
            d26_crossed = bool(st["d26_crossed"])
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
            log(f"[chunk {chunk_idx}] fresh start from the root "
                f"(replay to {REPLAY_TO}, then continue)")

        t_chunk = time.time()
        last_read_step = reads[-1]["step"] if reads else 0

        # ---- one chunk: steps until stop / cap / chunk-budget
        while stop is None:
            if step >= STEP_CAP:
                stop = {"kind": "cap", "step": step,
                        "note": "600-step cap reached with neither bar's "
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
            row = {"step": step, "chunk": chunk_idx,
                   "ce_batch": float(loss.item()), "cum_disp": cum_disp,
                   "step_disp": inc_disp, "preclip_gnorm": gnorm,
                   "lr_eff": LR_A2B, "x_md5": xh}
            journal.append(row)

            # ---- REPLAY GATE: identity vs opt1's committed a2b trajectory
            if step <= REPLAY_TO and str(step) != "0":
                srow = stored_traj.get(step)
                if srow is not None:
                    replay_diffs["ce"].append(
                        abs(row["ce_batch"] - srow["ce_batch"]))
                    replay_diffs["disp"].append(
                        abs(row["cum_disp"] - srow["cum_disp"]))
                    replay_diffs["gn"].append(
                        abs(row["preclip_gnorm"] - srow["preclip_gnorm"]))
                if step == REPLAY_TO:
                    mx = max(max(replay_diffs["ce"]),
                             max(replay_diffs["disp"]))
                    G_REPLAY = {
                        "replayed_steps": REPLAY_TO,
                        "max_abs_diff_ce": max(replay_diffs["ce"]),
                        "max_abs_diff_cum_disp": max(replay_diffs["disp"]),
                        "max_abs_diff_gnorm": max(replay_diffs["gn"]),
                        "max_abs_diff": mx, "bit_tol": G_BIT_TOL,
                        "tol": G_FALLBACK_TOL, "bit": bool(mx < G_BIT_TOL),
                        "pass": bool(mx < G_FALLBACK_TOL),
                        "flat_md5_at_replay_end": flat_md5(net),
                        "note": "the resumed state's identity check (Rule "
                                "12): deterministic replay of opt1's a2b "
                                "1..69 gated per step (ce, cum_disp, "
                                "preclip gnorm) vs the committed trajectory "
                                "— displacement continuity within float "
                                "tolerance; no opt1_a2b checkpoint exists "
                                "on disk, so this gate IS the identity "
                                "proof",
                    }
                    log(f"G_REPLAY (steps 1..{REPLAY_TO} vs opt1 a2b): "
                        f"max|dCE| {G_REPLAY['max_abs_diff_ce']:.2e} "
                        f"max|dD| {G_REPLAY['max_abs_diff_cum_disp']:.2e} "
                        f"max|dgn| {G_REPLAY['max_abs_diff_gnorm']:.2e}: "
                        + ("PASS" if G_REPLAY["pass"] else "FAIL")
                        + (" (bit)" if G_REPLAY["bit"] else ""))
                    if not G_REPLAY["pass"]:
                        raise RuntimeError("replay gate FAILED vs opt1's "
                                           "committed a2b trajectory")

            # ---- checkpoint read (cadence / forced at the 2.6 crossing)
            # first read at FIRST_READ (80); afterwards adaptive gaps from
            # the last read's g-12 (the replay itself performs no reads —
            # opt1's committed rows cover steps <= 40)
            due = (step > REPLAY_TO
                   and (step - max(last_read_step, REPLAY_TO))
                   >= read_gap())
            forced = (not d26_crossed) and cum_disp >= D_SPARE
            at_cap = step >= STEP_CAP
            if due or forced or (at_cap and step != last_read_step):
                if forced:
                    d26_crossed = True
                    d26_step, d26_disp = step, cum_disp
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
                log(f"  CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                    f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| "
                    f"{cum_disp:.4f} cos(g0) {cos_g0:+.3f}"
                    + ("  <= D=2.6 FORCED" if forced else ""))
                # ---- stop checks (whichever first)
                if gz["mean_pz"] <= SHUT_BAR:
                    prevr = reads[-2]
                    t_x = interp_cross(prevr["step"], prevr["gm12"],
                                       step, gz["mean_pz"], SHUT_BAR)
                    d_kill = interp_cross(prevr["step"], prevr["cum_disp"],
                                          step, cum_disp, t_x)
                    stop = {"kind": "kill", "step": step, "t_x": t_x,
                            "bracket": (prevr["step"], step),
                            "gm12_bracket": (prevr["gm12"], gz["mean_pz"]),
                            "D_kill_interp": d_kill,
                            "cum_disp_at_dead_read": cum_disp}
                    break
                if forced and gz["mean_pz"] > SHUT_BAR:
                    stop = {"kind": "spared", "step": step,
                            "gm12_at_gate": gz["mean_pz"],
                            "g0_at_gate": gz0["mean_pz"],
                            "ce_r_at_gate": ce_r,
                            "D_at_gate": cum_disp}
                    break

        # ---- chunk end: SAVE the full state (round-trip through disk)
        fp_md5 = flat_md5(net)
        payload = {"model": {k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items()},
                   "gen_state": gen.get_state().clone(),
                   "step": int(step), "chunk": int(chunk_idx),
                   "journal": journal, "reads": reads,
                   "d26_crossed": bool(d26_crossed),
                   "flat_md5": fp_md5,
                   "meta": {"experiment": "opt1b", "arm": "a2b_sgd_1e-2",
                            "lr": LR_A2B, "input_seed": FREEZE_SEED,
                            "base": f"runs/checkpoints/{ROOT_CK}",
                            "parent": "runs/opt1/metrics.json (a2b)"}}
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
                    "note": "600-step cap reached with neither bar's "
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
                        "hashes (= opt1's gate set); steps 11..69 carry "
                        "the per-step ce/displacement continuity gate; "
                        "all continuation-step md5s recorded in the "
                        "journal"}
    assert G_INPUTS["pass"], "replayed input stream diverged from e185/opt1"
    log(f"G_INPUTS: steps 1..10 md5s match e185's stored hashes: PASS")
    log(f"G_CHUNKS: {chunk_idx} chunks, {len(G_CHUNKS['round_trips'])} "
        f"disk round-trips, max chunk wall "
        f"{G_CHUNKS['max_chunk_wall_s']:.0f}s: "
        + ("PASS" if G_CHUNKS["pass"] else "FAIL"))

    # ---- the final checkpoint (provenance for any follow-up cell)
    fin_step = step
    fin_name = ("smoke_" if SMOKE else "") + \
        f"opt1b_a2b_sgd_1e-2_s{fin_step}"
    torch.save({"model": {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()},
                "meta": {"experiment": "opt1b", "arm": "a2b_sgd_1e-2",
                         "steps": int(fin_step), "lr": LR_A2B,
                         "input_seed": FREEZE_SEED,
                         "stop": stop["kind"],
                         "base": f"runs/checkpoints/{ROOT_CK}",
                         "parent": "runs/opt1/metrics.json (a2b)"}},
               CKPT_DIR / f"{fin_name}.pt")
    log(f"[ckpt] saved {fin_name}.pt (stop={stop['kind']} @ s{fin_step})")

    # =====================================================================
    # THE FACT-vs-D CURVE (opt1's committed a2b reads + this run's reads)
    # =====================================================================
    curve = [{"step": r["step"], "D": r["cum_disp"], "gm12": r["gm12"],
              "g0": r["g0"], "ce_r": r["ce_r"],
              "cos_delta_fact_g0": r["cos_delta_fact_g0"],
              "cos_delta_fact_m12": r["cos_delta_fact_m12"],
              "provenance": "opt1 (committed; replay bit-continuity-gated)"}
             for r in stored_ckpt]
    curve += [{"step": r["step"], "D": r["cum_disp"], "gm12": r["gm12"],
               "g0": r["g0"], "ce_r": r["ce_r"],
               "cos_delta_fact_g0": r["cos_delta_fact_g0"],
               "cos_delta_fact_m12": r["cos_delta_fact_m12"],
               "provenance": "opt1b (this run)"}
              for r in reads if r["step"] > REPLAY_TO]
    curve.sort(key=lambda r: r["step"])

    # =====================================================================
    # ADJUDICATION (registered clauses; composite order frozen:
    # SGD-KILLS-AT-GATE -> SGD-SPARED-AT-GATE -> CAP-NEITHER; no shopping)
    # =====================================================================
    kill_window_lo, kill_window_hi = KILL_WINDOW
    bars: dict = {}
    if stop["kind"] == "kill":
        d_kill = stop["D_kill_interp"]
        in_window = bool(kill_window_lo <= d_kill <= kill_window_hi)
        bars = {
            "SGD_KILLS_AT_GATE": {"fires": in_window,
                                  "D_kill_interp": d_kill,
                                  "window": list(KILL_WINDOW),
                                  "in_window": in_window,
                                  "t_x": stop["t_x"],
                                  "bracket": list(stop["bracket"]),
                                  "gm12_bracket": list(stop["gm12_bracket"])},
            "SGD_SPARED_AT_GATE": {"fires": False},
            "CAP_NEITHER": {"fires": False},
        }
        if in_window:
            verdict = "SGD-KILLS-AT-GATE"
            clause = (f"the fact died at interpolated displacement D_kill "
                      f"{d_kill:.3f} (t_x {stop['t_x']:.1f}, bracket "
                      f"{stop['bracket']}) — INSIDE the registered window "
                      f"[{kill_window_lo}, {kill_window_hi}] (2.49-2.84 "
                      f"± 15%): the displacement gate GENERALIZES across "
                      f"learned trajectory classes; the trajectory "
                      f"hypothesis (T137) keeps its two-class form "
                      f"(learned paths and static jumps).")
        else:
            verdict = "KILL-OUT-OF-WINDOW (no bar)"
            clause = (f"the fact died at interpolated displacement D_kill "
                      f"{d_kill:.3f} (t_x {stop['t_x']:.1f}) — OUTSIDE the "
                      f"registered window [{kill_window_lo}, "
                      f"{kill_window_hi}]: NEITHER bar fires (a kill "
                      f"outside the window is a graded outcome; the full "
                      "fact-vs-D curve is reported verbatim; no bar claim).")
    elif stop["kind"] == "spared":
        bars = {
            "SGD_KILLS_AT_GATE": {"fires": False},
            "SGD_SPARED_AT_GATE": {"fires": True,
                                   "gm12_at_gate": stop["gm12_at_gate"],
                                   "D_at_gate": stop["D_at_gate"],
                                   "step_at_gate": stop["step"]},
            "CAP_NEITHER": {"fires": False},
        }
        verdict = "SGD-SPARED-AT-GATE"
        clause = (f"the SGD trajectory PASSED D = 2.6 (reached "
                  f"{stop['D_at_gate']:.4f} at step {stop['step']}) with "
                  f"the fact ALIVE (g-12 {stop['gm12_at_gate']:.4f} > "
                  f"{SHUT_BAR}; g0 {stop['g0_at_gate']:.4f}, CE_R "
                  f"{stop['ce_r_at_gate']:.4f}) — the gate is "
                  f"TRAJECTORY-CLASS-TYPED (sign-normalized paths kill, "
                  f"raw-gradient paths are spared at the same "
                  f"displacement); the law gains a third clause and the "
                  f"paper's optimizer clause strengthens.")
    else:
        fin = journal[-1]
        # PROJECTED crossing (labeled EXTRAPOLATED; never adjudicates)
        tail = [r["step_disp"] for r in journal[-40:]]
        rate = sum(tail) / max(len(tail), 1)
        d_now = fin["cum_disp"]
        proj = (D_SPARE - d_now) / rate if rate > 0 else None
        g_last = reads[-1]["gm12"] if reads else None
        # linear projection of g-12 to 0.27 if it is falling
        g_proj = None
        if len(reads) >= 2 and reads[-1]["gm12"] < reads[-2]["gm12"]:
            s0, s1 = reads[-2]["step"], reads[-1]["step"]
            v0, v1 = reads[-2]["gm12"], reads[-1]["gm12"]
            if v0 > v1:
                g_proj = s1 + (v1 - SHUT_BAR) / (v0 - v1) * (s1 - s0)
        bars = {
            "SGD_KILLS_AT_GATE": {"fires": False},
            "SGD_SPARED_AT_GATE": {"fires": False},
            "CAP_NEITHER": {"fires": True},
            "projection": {"steady_disp_rate": rate,
                           "extrapolated_steps_to_D_2.6":
                               (fin["step"] + proj) if proj else None,
                           "extrapolated_kill_step": g_proj,
                           "label": "EXTRAPOLATED — projections never "
                                    "adjudicate"},
        }
        verdict = "CAP-NEITHER"
        clause = (f"neither fired within the {STEP_CAP}-step cap (final "
                  f"|d| {d_now:.4f}, last g-12 {g_last}); the fact-vs-D "
                  f"curve is reported and the projected crossings are "
                  f"labeled EXTRAPOLATED (never adjudicated): steady rate "
                  f"{rate:.5f}/step -> D=2.6 ~step "
                  f"{(fin['step'] + proj) if proj else float('nan'):.0f}"
                  + (f"; g-12 0.27 crossing ~step {g_proj:.0f} if the last "
                     "slope held" if g_proj else "") + ".")
    log("=" * 78)
    log(f"OPT1B VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # ---- W022/W023 alignment summary (co-read; report, no adjudication)
    my_reads = [r for r in reads]
    align_summary = None
    if my_reads:
        early = [r["cos_delta_fact_g0"] for r in my_reads
                 if r["cum_disp"] <= 1.0]
        late = [r["cos_delta_fact_g0"] for r in my_reads
                if r["cum_disp"] > 1.0]
        align_summary = {
            "mean_cos_g0_D_le_1": (sum(early) / len(early)) if early else None,
            "mean_cos_g0_D_gt_1": (sum(late) / len(late)) if late else None,
            "min_cos_g0": min(r["cos_delta_fact_g0"] for r in my_reads),
            "max_cos_g0": max(r["cos_delta_fact_g0"] for r in my_reads),
            "note": "the W022/W023 question: does SGD's cumulative "
                    "displacement drift more (more negative) or less "
                    "death-directed as D grows? W022's wash alignment read "
                    "~-0.44; opt1's arms sat at -0.015..-0.105",
        }

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "opt1b_sgd_kill",
        "date": common.now_iso(),
        "registration": ("the dispatch's registration IS the registration "
                         "(the three bars quoted verbatim in the module "
                         "docstring and in registered_prediction, frozen "
                         "before compute)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("opt1's owed discriminator: does opt1's A2b arm (SGD "
                     "lr 1e-2 on the licensed e185 wash cell) die at the "
                     "Adam displacement gate (D_kill within 2.49-2.84 "
                     "±15%), pass D=2.6 alive, or creep past the 600-step "
                     "cap? THE DIRECT SGD KILL, measured — the "
                     "extrapolated ~379-step clock replaced by a "
                     "trajectory"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "e185 arm C (CONTROL) / opt1 A2b VERBATIM — "
                             "the consolidated host fact under its "
                             "neutral/corpus wash",
            "batch": f"32 = {ANCH_BS} neutral-bank anchors (seed "
                     f"{E170_ANCHOR_SEED}, fixed) + {RAND_BS} random "
                     "corpus windows, full-token CE, clip 1.0",
            "targets": "TRUE (the real neutral wash — the optimizer is "
                       "the only delta vs the licensed cell)",
            "input_seed": FREEZE_SEED,
            "opt_spec": {"kind": "SGD", "lr": LR_A2B, "momentum": 0,
                         "weight_decay": 0.0, "clip": 1.0,
                         "lr_schedule": "constant"},
            "measure_light": "g-12 / g0 batteries + CE_R + alignment + "
                             "displacement + pre-clip grad norms (the "
                             "dispatch's read list; no census/deletions)",
        },
        "resume": {
            "opt1_a2b_checkpoints_on_disk": a2b_ckpts_on_disk,
            "method": ("NO a2b checkpoint exists (opt1 saved non-A0 arms "
                       "only at ladder-final steps; 69 is not on its "
                       "ladder) — the A2b state is re-established by "
                       "DETERMINISTIC REPLAY from the root: plain SGD "
                       "m=0/wd=0 is STATELESS and the stream is fixed by "
                       "torch.Generator().manual_seed(10902) with two "
                       "randint draws per step (aj then rj, opt1's "
                       "opt_wash order). The replay (steps 1..69) is "
                       "gated per step vs opt1's committed a2b trajectory "
                       "+ input md5s at steps 1..10 vs e185's stored "
                       "hashes — the resumed-state identity check"),
            "stream_continuity": ("the checkpoint's stream position "
                                  "carries in the GENERATOR STATE saved "
                                  "in chunk_state.pt; chunks round-trip "
                                  "the full state (model + generator + "
                                  "step + journal) through disk and each "
                                  "chunk LOADS the previous chunk's file "
                                  "(the cross-process path) — the input "
                                  "stream CONTINUES bit-identically "
                                  "(per-step md5s recorded; battery reads "
                                  "consume no RNG), never repeats"),
            "state_hash_chain": [c["flat_md5_at_end"] for c in chunks_prov],
            "flat_md5_at_replay_end": (G_REPLAY["flat_md5_at_replay_end"]
                                       if REPLAY_TO == 69 else None),
        },
        "arm": {
            "tag": "a2b_sgd_1e-2 (continued)",
            "parent": "runs/opt1/metrics.json arms.a2b_sgd_1e-2 "
                      "(69 steps, final |d| "
                      f"{a2b_tstar_opt1['final_cum_disp']:.4f}, rate "
                      f"{a2b_tstar_opt1['steady_disp_rate']:.5f}/step, "
                      "clock EXTRAPOLATED ~379 steps — never adjudicated)",
            "max_steps": STEP_CAP, "steps_ran": fin_step,
            "replay_steps": REPLAY_TO,
            "continuation_steps": fin_step - REPLAY_TO,
            "train_seconds": round(sum(c["wall_s"] for c in chunks_prov), 1),
            "time_cap_per_chunk": CHUNK_CAP,
            "seed": FREEZE_SEED,
            "traj": journal, "ckpt_table": reads,
            "zeph_violations": zeph_checks,
        },
        "chunks": chunks_prov,
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT, "G_REPLAY": (G_REPLAY
                                                  if REPLAY_TO == 69
                                                  else "trimmed (smoke)"),
                  "G_INPUTS": G_INPUTS, "G_DRAWFREE": G_DRAWFREE,
                  "G_CHUNKS": G_CHUNKS},
        "references": {
            "opt1": {"metrics": "runs/opt1/metrics.json",
                     "role": "the parent cell: A2b's committed trajectory "
                             "(the replay target), the Adam arms' "
                             "fact-vs-D curves (the overlay, plotted from "
                             "committed data), D_kill 2.4893 and the gate "
                             "band 2.49-2.84 (T139)"},
            "e185": {"metrics": "runs/e185/metrics.json",
                     "role": "the licensed cell + the stored input md5s "
                             "the replay re-derives"},
        },
        "fact_vs_D_curve": curve,
        "adam_overlay": {t: [{"step": r["step"], "D": r["cum_disp"],
                              "gm12": r["gm12"]}
                             for r in rows if r["step"] > 0]
                         for t, rows in adam_arms.items()},
        "coreads": {
            "alignment": align_summary,
            "ce_r": {"note": "does the organism keep learning? CE_R at "
                             "every checkpoint (rows in "
                             "fact_vs_D_curve / arm.ckpt_table)",
                     "root": root_cells["ce_r"],
                     "final": reads[-1]["ce_r"] if reads else None},
            "w022_w023": "alignment cos vs D: negative = death-aligned "
                         "(critic's convention); opt1's arms sat at "
                         "-0.015..-0.105, W022's wash read ~-0.44",
        },
        "adjudication": {
            "bars": bars,
            "verdict": verdict, "clause": clause,
            "stop": stop,
            "constants": {"D_spare_gate": D_SPARE,
                          "kill_window": list(KILL_WINDOW),
                          "adam_gate_band": list(ADAM_GATE_BAND),
                          "shut_bar": SHUT_BAR, "step_cap": STEP_CAP,
                          "D_kill_adam_A0": D_KILL_ADAM},
            "composite_order": "SGD-KILLS-AT-GATE -> SGD-SPARED-AT-GATE -> "
                               "CAP-NEITHER (frozen before compute)",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1, single seed lineage (10902, A2b's; the "
                            "replicate ladder is a separate dispatch "
                            "decision); ONE root, ONE input stream "
                            "(bit-gated vs opt1/e185); single family (the "
                            "e131 consolidated line); the continuation "
                            "differs from opt1's A2b in NOTHING (same "
                            "optimizer, same stream, same clip) — it is "
                            "the same training, resumed"),
            "extrapolation_free": ("every adjudication input is MEASURED "
                                   "inside this run's window: the kill "
                                   "(if any) is bracketed by reads and "
                                   "interpolated linearly-in-step; the "
                                   "spare is certified by a FORCED read at "
                                   "the measured 2.6-crossing step; only "
                                   "CAP-NEITHER carries projections, "
                                   "labeled EXTRAPOLATED and "
                                   "non-adjudicating by registration"),
            "float_texture": ("CPU fp32, this process, 4 threads (the "
                              "e185/opt1 reduction order); the replay "
                              "gate (max|dCE|/|dD| vs opt1's committed "
                              "trajectory) bounds cross-process float "
                              "drift explicitly"),
            "committed_data_reuse": ("opt1's a2b reads at steps "
                                     "{0,1,2,4,8,10,20,40} and the Adam "
                                     "arms' curves are REUSED from the "
                                     "committed runs/opt1/metrics.json "
                                     "(plotted, never rerun); this run's "
                                     "reads begin at step 80 + the forced "
                                     "gate read"),
            "no_guarantees": ("nothing was guaranteed ex ante: the SGD "
                              "path could kill at the gate, pass it "
                              "alive, or creep forever — the openness is "
                              "the point; the stop actually observed is "
                              f"'{stop['kind']}'"),
            "interpolation_texture": ("the kill's t_x/D_kill are "
                                      "linear-in-step interpolations "
                                      "inside a measured bracket whose "
                                      "width is reported; the bracket is "
                                      "the honest resolution limit"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": {
            fin_name: {"path": f"runs/checkpoints/{fin_name}.pt",
                       "arm": "a2b_sgd_1e-2", "steps": int(fin_step),
                       "stop": stop["kind"]},
            "chunk_state": {"path": "runs/opt1b/chunk_state.pt",
                            "note": "the resumable state (model + "
                                    "generator + step + journal)"},
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

    plot_fact_vs_D(rd / "opt1b_fact_vs_D.png", curve, adam_arms, verdict,
                   clause, stop, fin_step)
    plot_coreads(rd / "opt1b_coreads.png", curve, journal, reads,
                 chunks_prov, stop)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'opt1b_fact_vs_D.png'}, "
        f"{rd / 'opt1b_coreads.png'}, journal.jsonl, chunk_state.pt, "
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


def plot_fact_vs_D(path, curve, adam_arms, verdict, clause, stop, fin_step):
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    opt1_pts = [(c["D"], c["gm12"]) for c in curve
                if c["provenance"].startswith("opt1 ")]
    b_pts = [(c["D"], c["gm12"]) for c in curve
             if c["provenance"].startswith("opt1b")]
    if opt1_pts:
        ax.plot([p[0] for p in opt1_pts], [p[1] for p in opt1_pts], "o-",
                ms=5, lw=1.8, color="darkblue", alpha=0.9,
                label="SGD 1e-2 — opt1 committed (steps 0..40)")
    if b_pts:
        ax.plot([p[0] for p in b_pts], [p[1] for p in b_pts], "o-", ms=5,
                lw=1.8, color="royalblue", alpha=0.95,
                label=f"SGD 1e-2 — opt1b continuation (to s{fin_step})")
    for t, rows in adam_arms.items():
        col, lbl = ADAM_STYLE[t]
        pts = [(r["cum_disp"], r["gm12"]) for r in rows if r["step"] > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--", ms=5,
                lw=1.5, color=col, alpha=0.85, label=lbl)
    ax.axvspan(KILL_WINDOW[0], KILL_WINDOW[1], color="tab:purple",
               alpha=0.08, label=f"kill window [2.12, 3.27] (2.49-2.84 ±15%)")
    ax.axvline(D_SPARE, ls=":", lw=2.0, color="navy", alpha=0.8,
               label=f"D=2.6 spared gate (15% past the Adam gate)")
    for yv, col, lbl in ((0.50, "seagreen", "0.50 SPARE"),
                         (SHUT_BAR, "tab:purple", f"{SHUT_BAR} DISSOLVE")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    if stop["kind"] == "kill":
        ax.plot([stop["D_kill_interp"]], [SHUT_BAR], "*", ms=18,
                color="yellow", mec="k", zorder=6,
                label=f"SGD kill (interp D {stop['D_kill_interp']:.3f})")
    if stop["kind"] == "spared":
        ax.plot([stop["D_at_gate"]], [stop["gm12_at_gate"]], "*", ms=18,
                color="lime", mec="k", zorder=6,
                label=f"passed D=2.6 ALIVE (g-12 {stop['gm12_at_gate']:.3f})")
    ax.set_xlabel(r"cumulative displacement $\|\theta_t-\theta_0\|_2$ "
                  "(2.74M params)")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.2, loc="center left")
    ax.set_title("THE DIRECT SGD KILL — fact-vs-displacement, with opt1's "
                 "Adam arms overlaid (committed data)", fontsize=10)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"OPT1B VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.05
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=7.4, va="top",
                family="monospace")
        y -= 0.026
    y -= 0.02
    ax.text(0.02, y, "  step     D        g-12    g0     CE_R    cos(g0)  "
                     " provenance", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.028
    for c in curve:
        cos = c["cos_delta_fact_g0"]
        ax.text(0.02, y,
                f"  {c['step']:>5d}  {c['D']:7.4f}  {c['gm12']:.4f}  "
                f"{c['g0']:.4f}  {c['ce_r']:.4f}  "
                + (f"{cos:+.4f}" if cos is not None else "    n/a")
                + f"  {c['provenance'].split(' ')[0]}",
                fontsize=6.9, va="top", family="monospace",
                color=("royalblue" if c["provenance"].startswith("opt1b")
                       else "darkblue"))
        y -= 0.026
    fig.suptitle("OPT1B — THE DIRECT SGD KILL: opt1's A2b continued to the "
                 "measured kill / D=2.6 / 600-step cap", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coreads(path, curve, journal, reads, chunks_prov, stop):
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    # (0,0) g-12 vs step with chunk boundaries
    ax = axes[0, 0]
    steps = [c["step"] for c in curve]
    ax.plot(steps, [c["gm12"] for c in curve], "o-", ms=5, lw=1.8,
            color="royalblue", label="g-12 (opt1 rows + opt1b reads)")
    ax.plot([r["step"] for r in reads], [r["gm12"] for r in reads], "o",
            ms=7, mfc="none", mec="navy", label="opt1b reads")
    for c in chunks_prov[:-1]:
        ax.axvline(c["step_to"], ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.axhline(SHUT_BAR, ls="--", lw=1.2, color="tab:purple",
               label=f"{SHUT_BAR} DISSOLVE")
    ax.set_xlabel("wash step (chunk boundaries dotted)")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("the kill clock in STEPS (chunked, ckpt-resumable)", fontsize=10)

    # (0,1) D(t) vs step + the gates
    ax = axes[0, 1]
    ax.plot([r["step"] for r in journal], [r["cum_disp"] for r in journal],
            "-", lw=1.7, color="royalblue", label="cumulative |d| (SGD 1e-2)")
    ax.axhline(D_SPARE, ls=":", lw=1.8, color="navy", alpha=0.9,
               label=f"D=2.6 spared gate")
    ax.axhspan(ADAM_GATE_BAND[0], ADAM_GATE_BAND[1], color="tab:purple",
               alpha=0.10, label="Adam kill band 2.49-2.84 (opt1)")
    ax.axhline(D_KILL_ADAM, ls="--", lw=1.0, color="crimson", alpha=0.7,
               label="A0 D_kill 2.489")
    for c in chunks_prov[:-1]:
        ax.axvline(c["step_to"], ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"$\|\theta_t-\theta_0\|_2$")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("displacement vs step (the measured trajectory)", fontsize=10)

    # (1,0) alignment vs D (the W022/W023 co-read)
    ax = axes[1, 0]
    rows = [c for c in curve if c["cos_delta_fact_g0"] is not None]
    ax.plot([c["D"] for c in rows], [c["cos_delta_fact_g0"] for c in rows],
            "o-", ms=5, lw=1.8, color="teal", label="cos(Δθ, ∇g0)")
    ax.plot([c["D"] for c in rows], [c["cos_delta_fact_m12"] for c in rows],
            "s--", ms=4.5, lw=1.5, color="olive", alpha=0.8,
            label="cos(Δθ, ∇m12)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(-0.44, ls="--", lw=1.0, color="gray", alpha=0.8,
               label="W022 wash read ~-0.44")
    ax.set_xlabel("cumulative displacement D")
    ax.set_ylabel("alignment (negative = death-aligned)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("THE W022/W023 CO-READ — does SGD's path drift "
                 "death-directed as D grows?", fontsize=9.5)

    # (1,1) CE_R vs D (does the organism keep learning?)
    ax = axes[1, 1]
    ax.plot([c["D"] for c in curve], [c["ce_r"] for c in curve], "o-", ms=5,
            lw=1.8, color="saddlebrown", label="CE_R (organism health)")
    ax.axhline(curve[0]["ce_r"], ls="--", lw=1.0, color="gray",
               alpha=0.8, label=f"root CE_R {curve[0]['ce_r']:.3f}")
    ax.set_xlabel("cumulative displacement D")
    ax.set_ylabel("CE_R")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("does the organism keep learning under the SGD wash?",
                 fontsize=10)

    fig.suptitle("OPT1B — co-reads: the clock, the displacement, the "
                 "alignment, the organism", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
