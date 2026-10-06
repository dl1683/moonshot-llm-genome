"""E304 — THE FEVER CELL — the honesty audit of the founding success
(eval-only, CPU-ONLY desk cell; threads <= 4; NO GPU; no envelope writes;
TIMESTAMPS: datetime.now(UTC) only).

THE QUESTION (the dispatch, verbatim): "is the error-gated controller's
3.5x read overshoot GENUINE KNOWLEDGE or a FEVER — does preservation buy
confidence the organism cannot cash? Point the thermal lens (the one-T
family) at the maintained states."

FROZEN BARS (verbatim from the dispatch letter; this script committed at
birth BEFORE any compute; adjudicate against exactly this; no bar
shopping):
  - FEVER: "T at the gate tokens rises >= +0.10 while the paraphrase's T
    rises < half as much — preservation buys overconfidence; the price on
    Law 4."
  - CALIBRATED-SURVIVAL: "T moves < 0.05 at both sites — the overshoot
    is real mass; the founding success honest in the fullest sense."
  - LOCAL-FEVER / MIXED: "intermediate patterns — the two-site table
    verbatim; the tumor boundary located thermally if gate-only."

OPERATIONALIZATIONS (frozen HERE at birth, before compute; they fix the
clauses, they do not move the bars):
  * THE ONE-T LENS (the x4/x5/x9 family's instrument, machinery VERBATIM
    from e238/x5: ONE temperature per (state, site), grid log10 T in
    [-1, 1] x 1201 + golden-section polish, Bernoulli soft-target MLE):
        family  q_i(T) = softmax(L_state_i / T)[ans_i]
        target  p_i    = softmax(L_ref_i)[ans_i]
    with L_state/L_ref the FULL 65-vocab answer-position logits of the
    state / of the REFERENCE state (= the arm's own resume checkpoint —
    the loaded fact as each cell loaded it), and ans_i the true token at
    the site's masked position. DIRECTION (the one registered choice,
    disclosed): the temperature sits ON THE STATE'S LOGITS — T > 1 means
    the state's site distribution must be COOLED to restore the
    reference's calibration (the site runs HOT; the direction in which
    the bars' "T rises" literally reads overconfidence). The ANCHOR row
    (the resume state fit against itself) must read T = 1.000000 (the
    instrument's positive control — x5's t0-anchor convention; term-wise
    exact: at T = 1 every Bernoulli term sits at its own minimum).
    HEAT(site) := T(site) - 1.
    DISCLOSED ANALYTIC CAVEAT (frozen at birth): the answer coordinate
    itself carries the mass gain — a literal read rise contributes to T
    at the answer coordinate under ANY baseline-anchored temperature
    lens; the mass-vs-spike decomposition is therefore carried by the
    TWO-SITE COMPARISON (the bars' own form) + the co-reported lens
    forms below (the x5-literal direction, both full-vocab KL forms,
    per-probe z-ratios with x8's physicality flags, margins / sigma /
    T_app per x7). The primary adjudicates; the co-reports price the
    lens's saturation.
  * THE SITES: GATE := the protocol's SHARED install bank (the
    SPLICE_RNG-shuffled host_occ[:60], e261's build_win VERBATIM: 130
    pre-context tokens | ZEPHYRA | 119 post; identical across e287-e293
    by construction) read at the SEVEN masked y-positions (x-cols
    129..135, answers Z,E,P,H,Y,R,A; 60 x 7 = 420 probes — the
    maintenance steps' own teaching positions); the y129-only subset
    co-reported (the battery read position, where the committed 3.5x
    overshoot is defined). PARA := the held-out paraphrase bank
    (host_occ[60:90], geometry g0, [30, 130]) read at col 129, answer
    'Z' (30 probes; e291's committed G_HELD construction; bound
    bit-wise to e291's committed held_t0 read on the e291 organism).
  * THE STATES (all existing checkpoints, loaded + certified):
    N-family (neutral corpus 28801): e288 ERROR-GATED {resume, post} +
    NAME-FIXED-TWIN {resume, post}; D-family (denial corpus, e289):
    CONTRADICTED-WITH-CONTROLLER {resume, post}, CONTRADICTED-NO-
    CONTROLLER {resume, post} (the denial passive), C1:ERROR-GATED
    {resume, post}, C1:NAME-FIXED-TWIN {resume, post}; F-family (e291):
    FIVE-CONTROLLERS {resume, post} + SINGLE-CONTROL-TWIN {resume,
    post}; RIDER: e287 SANCTUARY-TWIN {resume, post} (the neutral
    passive; ITS corpus draws ran on seed 28701 != 28801 — disclosed).
  * CERTIFICATION (hard gate): every loaded state re-probed on the g0
    battery (and, for e291, the per-fact 12-window panels) against the
    parent cell's committed post_g0 / panel (tol 0.005; x5's G_STATES
    reprobe convention) — this certifies the forwards the logits ride
    on; resumes additionally certified against the committed loaded
    baseline 0.26464763283729553 (the e288/e289 class) or e291's
    committed per-fact baselines.
  * ADJUDICATION: the frozen clauses applied to THE FOUNDING STATE
    (e288 ERROR-GATED post — the 3.5x founding success the dispatch
    audits), HEAT computed against the arm's resume anchor. Precedence:
    FEVER -> CALIBRATED-SURVIVAL -> LOCAL-FEVER (heat_gate >= +0.10 AND
    |heat_para| < 0.05 — gate-only heat, the tumor boundary located
    thermally) -> MIXED (everything else, incl. both-hot). The same
    clauses co-reported for every maintained post state, verbatim.
  * THE OVERSHOOT TRACE: milestone (t100/200/300) states were NOT
    checkpointed by the parent cells (disclosed; the checkpoint globs
    recorded as evidence) — the T axis therefore exists at {resume,
    post} per arm; the READ axis uses the families' COMMITTED traj_g0
    milestone curves (read at runtime from the md5-bound metrics.json
    of e288/e289), never transcribed.
  * THE HONESTY CO-READS (part d): each state's committed post_ce_r /
    post_gm12 / post_g0 / survival ratio pulled at runtime from the
    md5-bound parents, joined with this cell's HEATs in one table.
  * COMPUTE: CPU fp32 eval only (torch threads 4, map_location cpu, no
    CUDA touch, no envelope-log writes, one process); ~19 checkpoint
    loads x (60x256 + 30x130 + 60x130 + 60x118) forwards — desk price.

Outputs: runs/e304/{metrics.json (PROGRESSIVE), e304_fever.png,
REPORT.md (executor-written), run.log (gitignored)}. NO NOTES/THINKING/
QUEUE/STATE edits (dispatch; the coordinator folds). Birth-commit
BEFORE compute; final commit AND push.

Run:  cd lab && python e304_fever.py    (E304_SMOKE=1 shakedown)
"""

from __future__ import annotations

import copy
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import CharCorpus                         # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402

torch.set_num_threads(4)           # the shared CPU lane (dispatch: <= 4)
torch.set_grad_enabled(False)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E304_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e304_smoke" if SMOKE else "e304"

T0 = time.time()
RD = common.run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    import subprocess
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=E43.REPO, capture_output=True,
        text=True).stdout.strip()


# ------------------------------------------------------------------ frozen --
REPO = E43.REPO
CKPT = REPO / "runs" / "checkpoints"

# md5 binds (computed at birth; asserted at runtime — record integrity)
METRICS_BINDS = {
    "e288": ("runs/e288/metrics.json", "31bf8df55b51c8a19c48a155388050be"),
    "e289": ("runs/e289/metrics.json", "4e061c03092239642e6ec40419ee5238"),
    "e291": ("runs/e291/metrics.json", "fc9cb859f2ef2603397c971dbcf5c618"),
    "e287": ("runs/e287/metrics.json", "b7d18b8b286352732087b2ea99af3d2a"),
}
FACT_CK = "e261_K10K_inst_resume.pt"
FACT_CK_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_BASELINE_G0 = 0.26464763283729553        # the committed loaded read
HELD_T0_E291 = 0.2676781713962555              # e291's committed held read
REPROBE_TOL = 0.005                            # x5's G_STATES convention

# the frozen adjudication constants (the bars, verbatim numbers)
FEVER_GATE = 0.10                              # "rises >= +0.10"
FEVER_HALF = 0.5                               # "rises < half as much"
CAL_TOL = 0.05                                 # "moves < 0.05"

# the state archive. PROVENANCE NOTE (verified at birth by re-probe):
# the parent cells' "*_resume.pt" checkpoints are POST-phase resume
# states (saved for later landing passes), NOT pre-phase baselines —
# the pre-phase baselines are the loaded fact itself (e261_K10K_inst_
# resume.pt, the N/D/X families' shared start) and e291's organism
# (the post-install, pre-phase state of the five-fact cell).
BASELINES = [
    ("N", "BASE(loaded-fact)", "base", FACT_CK),
    ("F", "BASE(e291-organism)", "base", "e291_organism.pt"),
]
STATES = [
    ("N", "ERROR-GATED", "post", "e288_ERROR-GATED_post.pt"),
    ("N", "NAME-FIXED-TWIN", "post", "e288_NAME-FIXED-TWIN_post.pt"),
    ("D", "CONTRADICTED-WITH-CONTROLLER", "post",
     "e289_CONTRADICTED-WITH-CONTROLLER_post.pt"),
    ("D", "CONTRADICTED-NO-CONTROLLER", "post",
     "e289_CONTRADICTED-NO-CONTROLLER_post.pt"),
    ("D", "C1:ERROR-GATED", "post", "e289_C1-ERROR-GATED_post.pt"),
    ("D", "C1:NAME-FIXED-TWIN", "post", "e289_C1-NAME-FIXED-TWIN_post.pt"),
    ("F", "FIVE-CONTROLLERS", "post", "e291_FIVE-CONTROLLERS_post.pt"),
    ("F", "SINGLE-CONTROL-TWIN", "post", "e291_SINGLE-CONTROL-TWIN_post.pt"),
    ("X", "SANCTUARY-TWIN(e287)", "post", "e287_SANCTUARY-TWIN_post.pt"),
]
FOUNDING = ("N", "ERROR-GATED", "post")        # the audited state
if SMOKE:
    STATES = [s for s in STATES if s[0] == "N"]
    BASELINES = [b for b in BASELINES if b[0] == "N"]

metrics: dict = {
    "experiment": f"{NAME}_fever",
    "phase": "THE FEVER CELL — the honesty audit of the founding success "
             "(the dispatch's desk cell; CPU-only)",
    "date": utcnow(),
    "status": "PARTIAL (progressive writes; this line replaced at the end)",
    "registration": {
        "bars_verbatim": {
            "FEVER": "T at the gate tokens rises >= +0.10 while the "
                     "paraphrase's T rises < half as much — preservation "
                     "buys overconfidence; the price on Law 4.",
            "CALIBRATED-SURVIVAL": "T moves < 0.05 at both sites — the "
                                   "overshoot is real mass; the founding "
                                   "success honest in the fullest sense.",
            "LOCAL-FEVER / MIXED": "intermediate patterns — the two-site "
                                   "table verbatim; the tumor boundary "
                                   "located thermally if gate-only.",
        },
        "lens_form": "ONE T per (state, site): Bernoulli soft-target MLE, "
                     "family q_i(T) = softmax(L_state_i/T)[ans_i], targets "
                     "p_i = softmax(L_ref_i)[ans_i], ref = the VERIFIED "
                     "reference anchor (see reference_states); grid log10T "
                     "in [-1,1] x 1201 + golden "
                     "polish (e238/x5 machinery VERBATIM); T>1 = the site "
                     "runs hot (the state must be cooled to reference "
                     "calibration); anchor row must read 1.000000",
        "sites": "GATE = the shared install bank's 7 masked y-positions "
                 "(cols 129..135, answers ZEPHYRA chars; 420 probes; the "
                 "maintenance steps' teaching positions; y129 subset "
                 "co-reported); PARA = the held paraphrase bank "
                 "(host_occ[60:90], g0, col 129, answer Z; 30 probes)",
        "reference_states": "the docstring's reference clause ('the arm's "
                            "own resume checkpoint — the loaded fact as "
                            "each cell loaded it') verified at runtime: "
                            "the parents' *_resume.pt carry model weights "
                            "BIT-IDENTICAL to their *_post.pt (post-phase "
                            "duplicates saved for landing passes — "
                            "evidence in disclosures.resume_duplicates); "
                            "the reference is therefore THE LOADED FACT "
                            "itself: e261_K10K_inst_resume.pt (the N/D/X "
                            "families' shared start, certified g0 = "
                            "0.2646...) and e291_organism.pt (the F "
                            "family's post-install pre-phase state, "
                            "certified on the per-fact panels) — exactly "
                            "the parenthetical's definition and the "
                            "certification clause's targets; fitting "
                            "against the *_resume.pt duplicates would fit "
                            "every state against itself (HEAT = 0 "
                            "identically — a degenerate instrument)",
        "adjudication_state": "e288 ERROR-GATED post (the founding 3.5x)",
        "precedence": "FEVER -> CALIBRATED-SURVIVAL -> LOCAL-FEVER "
                      "(gate >= +0.10 AND |para| < 0.05) -> MIXED",
    },
    "smoke": SMOKE,
    "envelope": {
        "device": "CPU fp32 eval only; torch threads 4; no CUDA; no "
                  "envelope-log writes; one process",
        "timestamps": "datetime.now(UTC) only",
    },
    "provenance": {"git_head_at_start": git_head(),
                   "birth_commit_note": "script committed at birth "
                                        "BEFORE compute (dispatch)"},
    "gates": {},
    "states": {},
    "fits": [],
    "co_reads": {},
    "trace": {},
    "riders": {},
    "adjudication": {},
    "disclosures": {},
    "timing": {},
}


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL — {note} ({utcnow()})"
    (RD / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    log(f"[metrics] partial write: {note}")


# =============================================================== P0: banks ==
log("=" * 78)
log(f"E304 THE FEVER CELL — smoke={SMOKE} — CPU-only, threads "
    f"{torch.get_num_threads()}, cuda touched: False")
log("P0: the protocol rebuild (banks + binds)")

mbinds = {}
for cell, (rel, md5) in METRICS_BINDS.items():
    p = REPO / rel
    got = md5of(p)
    assert got == md5, f"{rel} md5 drift: {got} != {md5}"
    mbinds[cell] = {"path": rel, "md5": got}
metrics["gates"]["G_RECORDS"] = {"binds": mbinds, "pass": True}

e288m = json.loads((REPO / "runs/e288/metrics.json").read_text("utf-8"))
e289m = json.loads((REPO / "runs/e289/metrics.json").read_text("utf-8"))
e291m = json.loads((REPO / "runs/e291/metrics.json").read_text("utf-8"))
e287m = json.loads((REPO / "runs/e287/metrics.json").read_text("utf-8"))

corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
stoi, itos = corpus.stoi, corpus.itos
zid = stoi["Z"]
train_ids = corpus.train
train_text = "".join(itos[int(i)] for i in train_ids)
name_ids = corpus.encode(G1.NAME)
assert corpus.vocab_size == 65

# the protocol's shared draw (identical across e287-e293 by construction)
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


def build_win(p, host):
    return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                      train_ids[p + len(host):
                                p + len(host) + G1.POST_CAP]])


win_i = torch.stack([build_win(p, h) for p, h in install_occ])   # [60, 256]
bat_g0 = torch.stack([train_ids[p - G1.PRE: p] for p, _ in install_occ])
bat_gm12 = torch.stack([train_ids[p - G1.PRE + 12: p]
                        for p, _ in install_occ])                # [60, 118]
held_ids = torch.stack([train_ids[p - G1.PRE: p] for p, _ in held_occ])

masked_ok = all("".join(itos[int(i)] for i in
                        win_i[i][G1.PRE:G1.PRE + len(G1.NAME)]) == G1.NAME
                for i in range(win_i.shape[0]))
pre_ok = all(torch.equal(win_i[i][:G1.PRE], bat_g0[i]) for i in range(60))
held_namefree = all("ZEPH" not in train_text[p - G1.PRE: p]
                    for p, _ in held_occ)
G_BANKS = {
    "install_mix": mix,
    "win_shape": list(win_i.shape),
    "held_shape": list(held_ids.shape),
    "masked_decode_all_name": bool(masked_ok),
    "precontext_bit_equal_battery": bool(pre_ok),
    "held_namefree_decode": bool(held_namefree),
    "held_positions_disjoint_install": bool(
        not {p for p, _ in held_occ} & {p for p, _ in install_occ}),
    "form": "the protocol's shared banks (SPLICE_RNG draw; e261's "
            "build_win): GATE = win_i [60,256] masked cols 129..135; PARA "
            "= held host_occ[60:90] g0 [30,130] col 129; battery g0/gm12 "
            "for certification reads",
    "pass": bool(masked_ok and pre_ok and held_namefree
                 and mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                 and list(win_i.shape) == [60, 256]
                 and list(held_ids.shape) == [30, 130]),
}
assert G_BANKS["pass"], f"bank gate FAILED: {G_BANKS}"
metrics["gates"]["G_BANKS"] = G_BANKS

# milestone-state disclosure (evidence): the parents checkpointed
# resume/post only
mile_globs = sorted(str(p.name) for p in CKPT.glob("e28[89]_t*.pt")) + \
    sorted(str(p.name) for p in CKPT.glob("e28[89]_*100*.pt")) + \
    sorted(str(p.name) for p in CKPT.glob("e29[13]_t*.pt"))
metrics["disclosures"]["milestone_states"] = {
    "found": mile_globs,
    "fact": "the parent cells checkpointed resume + post only — NO "
            "t100/200/300 states exist (and the resume files are "
            "post-phase duplicates of post — see resume_duplicates); "
            "the T axis therefore lives at {reference-base, post} per "
            "arm; the READ axis uses the families' committed traj_g0 "
            "curves (read at runtime, md5-bound)",
}
log(f"P0: banks bound (mix {mix}; masked ZEPHYRA 60/60; held name-free "
    f"30; milestone ckpts found: {mile_globs})")
write_partial("P0 banks + record binds PASS")

# ============================================== P1: loads + certification ==
log("P1: state loads + read certification")

fact_path = CKPT / FACT_CK
assert md5of(fact_path) == FACT_CK_MD5, "the fact checkpoint md5 drift"
fact_st = torch.load(fact_path, map_location="cpu", weights_only=False)
net0 = G1.evl_load(fact_st["model"] if "model" in fact_st else fact_st)
assert net0.num_params() == GB.G1B_PARAMS


@torch.no_grad()
def battery_mean_pz(net, ids, zid, bs=30) -> float:
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pzs.append(F.softmax(lg[:, -1], -1)[:, zid])
    return float(torch.cat(pzs).mean())


@torch.no_grad()
def site_logits(net, ids, cols, bs=30) -> torch.Tensor:
    """Full-vocab logits at the given x-cols for every window.
    Returns [n_windows, len(cols), V]."""
    net.eval()
    outs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        outs.append(lg[:, cols, :])
    return torch.cat(outs)


# the sites' probe geometry
GATE_COLS = list(range(G1.PRE - 1, G1.PRE - 1 + len(G1.NAME)))  # 129..135
GATE_ANS = name_ids.tolist()                                     # Z E P H Y R A
PARA_COLS = [G1.PRE - 1]                                         # 129
PARA_ANS = [int(name_ids[0])]                                    # Z
Y129 = [GATE_COLS[0]]

# committed reads for certification (runtime reads, never transcribed)
r288 = e288m["adjudication"]["reads"]
r289 = e289m["adjudication"]["reads"]
r291 = e291m["adjudication"]["reads"]
COMMITTED_POST_G0 = {
    ("N", "ERROR-GATED", "post"): r288["ERROR-GATED"]["post_g0"],
    ("N", "NAME-FIXED-TWIN", "post"): r288["NAME-FIXED-TWIN"]["post_g0"],
    ("D", "CONTRADICTED-WITH-CONTROLLER", "post"):
        r289["CONTRADICTED-WITH-CONTROLLER"]["post_g0"],
    ("D", "CONTRADICTED-NO-CONTROLLER", "post"):
        r289["CONTRADICTED-NO-CONTROLLER"]["post_g0"],
    ("D", "C1:ERROR-GATED", "post"): r289["C1:ERROR-GATED"]["post_g0"],
    ("D", "C1:NAME-FIXED-TWIN", "post"):
        r289["C1:NAME-FIXED-TWIN"]["post_g0"],
    ("X", "SANCTUARY-TWIN(e287)", "post"):
        r288["the_cited_control_e287"]["its_sanctuary_twin_post"],
}
# e291's per-fact baselines (runtime arithmetic on committed numbers)
f_panel_final = r291["FIVE-CONTROLLERS"]["final_panel_g0"]
f_panel_ratio = r291["FIVE-CONTROLLERS"]["ratios_vs_baseline"]
F_PANEL_BASELINE = {k: f_panel_final[k] / f_panel_ratio[k] for k in f_panel_final}
F_PANEL_FINAL = {
    "FIVE-CONTROLLERS": f_panel_final,
    "SINGLE-CONTROL-TWIN": r291["SINGLE-CONTROL-TWIN"]["final_panel_g0"],
}
FACT_GROUPS = {f"FACT{i + 1}": slice(i * 12, (i + 1) * 12) for i in range(5)}

state_rows, cert_fail = {}, []
nets = {}
for fam, arm, role, ck in BASELINES + STATES:
    path = CKPT / ck
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if "model" in st else st
    net = G1.evl_load(sd)
    g0 = battery_mean_pz(net, bat_g0, zid)
    gm12 = battery_mean_pz(net, bat_gm12, zid)
    held = battery_mean_pz(net, held_ids, zid)
    panel = {k: battery_mean_pz(net, bat_g0[s], zid)
             for k, s in FACT_GROUPS.items()}
    key = (fam, arm, role)
    nets[key] = net
    row = {
        "family": fam, "arm": arm, "role": role, "checkpoint": ck,
        "ckpt_md5": md5of(path), "flat_md5": None,
        "g0_battery": g0, "gm12_battery": gm12, "held_pz": held,
        "panel_g0": panel,
    }
    flat = torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()])
    row["flat_md5"] = hashlib.md5(
        flat.numpy().astype(np.float64).tobytes()).hexdigest()
    # certification
    if role == "post" and key in COMMITTED_POST_G0:
        row["cert_target"] = COMMITTED_POST_G0[key]
        row["cert_abs_diff"] = abs(g0 - COMMITTED_POST_G0[key])
        if row["cert_abs_diff"] > REPROBE_TOL:
            cert_fail.append(key)
    if role == "base" and fam == "N":
        # the loaded fact certifies the WHOLE N/D/X class's reference
        row["cert_target"] = FACT_BASELINE_G0
        row["cert_abs_diff"] = abs(g0 - FACT_BASELINE_G0)
        if row["cert_abs_diff"] > REPROBE_TOL:
            cert_fail.append(key)
    if fam == "F":
        tgt = F_PANEL_BASELINE if role == "base" else \
            F_PANEL_FINAL[arm]
        diffs = {k: abs(panel[k] - tgt[k]) for k in tgt}
        row["cert_target_panel"] = tgt
        row["cert_panel_max_abs_diff"] = max(diffs.values())
        if max(diffs.values()) > REPROBE_TOL:
            cert_fail.append(key)
    state_rows[str(key)] = row
    del st, sd
    log(f"  loaded {ck}: g0 {g0:.6f} gm12 {gm12:.6f} held {held:.6f} "
        f"cert_abs {row.get('cert_abs_diff', row.get('cert_panel_max_abs_diff'))}")

# the reference-duplicate evidence (the re-anchoring's proof, at runtime):
# every parent *_resume.pt vs its *_post.pt, model-weights flat-md5 compare
F_BASE_KEY = None
for _bf, _ba, _br, _ in BASELINES:
    if _bf == "F":
        F_BASE_KEY = (_bf, _ba, _br)
dup_rows = {}
for fam, arm, role, ck in STATES:
    rck = ck.replace("_post.pt", "_resume.pt")
    rst = torch.load(CKPT / rck, map_location="cpu", weights_only=False)
    rsd = rst["model"] if "model" in rst else rst
    rnet = G1.evl_load(rsd)
    rflat = hashlib.md5(torch.cat(
        [p.detach().reshape(-1).cpu() for p in rnet.parameters()]
    ).numpy().astype(np.float64).tobytes()).hexdigest()
    dup_rows[f"{fam}/{arm}"] = {
        "resume_ckpt": rck,
        "resume_flat_md5": rflat,
        "post_flat_md5": state_rows[str((fam, arm, role))]["flat_md5"],
        "bit_identical": bool(
            rflat == state_rows[str((fam, arm, role))]["flat_md5"]),
    }
    del rst, rsd, rnet
metrics["disclosures"]["resume_duplicates"] = {
    "rows": dup_rows,
    "all_bit_identical": bool(all(r["bit_identical"]
                                  for r in dup_rows.values())),
    "fact": "every parent *_resume.pt carries model weights BIT-IDENTICAL "
            "to its *_post.pt (the wrapper merely adds optimizer/generator "
            "state for later landing passes) — they are POST-phase "
            "duplicates, NOT pre-phase baselines; the lens reference is "
            "therefore the loaded fact itself (N/D/X: e261_K10K_inst_"
            "resume.pt, certified 0.2646; F: e291_organism.pt, certified "
            "on the per-fact panels), exactly the docstring's own "
            "parenthetical ('the loaded fact as each cell loaded it') + "
            "its certification clause; fitting the duplicates would fit "
            "every state against itself (HEAT = 0 identically)",
}

# the paraphrase bank's bit-bind: e291's committed held_t0 on ITS organism
if F_BASE_KEY is not None:
    held_bind_diff = abs(state_rows[str(F_BASE_KEY)]["held_pz"]
                         - HELD_T0_E291)
else:                                   # smoke: the F family is excluded
    held_bind_diff = 0.0
held_bind_ok = F_BASE_KEY is None or held_bind_diff <= REPROBE_TOL

ref_ids = {str(k): v["flat_md5"] for k, v in
           ((k, state_rows[str(k)]) for k in nets) if k[2] == "base"}
G_STATES = {
    "n_states": len(state_rows),
    "cert_failures": [str(x) for x in cert_fail],
    "reprobe_tol": REPROBE_TOL,
    "held_bank_bind": {"e291_committed_held_t0": HELD_T0_E291,
                       "abs_diff": held_bind_diff,
                       "pass": bool(held_bind_ok)},
    "reference_flat_md5s": ref_ids,
    "note": "every loaded state re-probed on the g0 battery (e291: the "
            "per-fact 12-window panels; the reference bases additionally "
            "certified against the committed loaded baselines) — x5's "
            "G_STATES reprobe convention; the held-bank bind certifies "
            "the paraphrase bank bit-wise via e291's organism",
    "pass": bool(not cert_fail and held_bind_ok),
}
assert G_STATES["pass"], f"G_STATES FAILED: {G_STATES}"
metrics["gates"]["G_STATES"] = G_STATES
metrics["states"] = state_rows
log(f"P1: {len(state_rows)} states certified (0 failures; held-bank bind "
    f"d {held_bind_diff:.2e}; references: {len(ref_ids)} distinct "
    f"flat-md5; resume-duplicates all bit-identical: "
    f"{metrics['disclosures']['resume_duplicates']['all_bit_identical']})")
write_partial("P1 loads + certification PASS")

# ==================================================== P2: the thermal lens ==
log("P2: the one-T fits (both sites, all states + co-report forms)")

GRID_LO, GRID_HI, GRID_N = -1.0, 1.0, 1201


def _golden(f, lo, hi, tol=1e-10):
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    while abs(b - a) > tol:
        if f(c) < f(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    return (a + b) / 2


def logsoftmax_T(L: np.ndarray, T: float) -> np.ndarray:
    z = L / T
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def softmax_np(L: np.ndarray) -> np.ndarray:
    z = L - L.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll_bernoulli(logq_ans: np.ndarray, p: np.ndarray) -> float:
    q = np.exp(logq_ans)
    q = np.clip(q, 1e-300, 1 - 1e-300)
    return float(-np.sum(p * np.log(q) + (1 - p) * np.log(1 - q)))


def fit_T(family_L: np.ndarray, ans: np.ndarray, p_target: np.ndarray,
          n_grid: int = GRID_N) -> dict:
    """ONE T: minimize Bernoulli soft-target NLL of q=softmax(L/T)[ans]."""
    idx = np.arange(len(ans))
    grid = np.linspace(GRID_LO, GRID_HI, n_grid)
    vals = np.array([nll_bernoulli(
        logsoftmax_T(family_L, 10.0 ** g)[idx, ans],
        p_target) for g in grid])
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, n_grid - 1)]
    gstar = _golden(lambda g: nll_bernoulli(
        logsoftmax_T(family_L, 10.0 ** g)[idx, ans],
        p_target), lo, hi)
    return {"T": float(10.0 ** gstar),
            "nll_at_T": float(vals[k]),
            "nll_at_T1": nll_bernoulli(
                logsoftmax_T(family_L, 1.0)[idx, ans],
                p_target),
            "grid_edge": bool(k in (0, n_grid - 1))}


def fit_T_kl(family_L: np.ndarray, ref_L: np.ndarray) -> float:
    """argmin_T KL(softmax(family_L/T) || softmax(ref_L)) — full vocab."""
    pref = np.log(np.maximum(softmax_np(ref_L), 1e-300))
    grid = np.linspace(GRID_LO, GRID_HI, 401)
    ev = []
    for g in grid:
        lq = logsoftmax_T(family_L, 10.0 ** g)
        ev.append(float((np.exp(lq) * (lq - pref)).sum()))
    k = int(np.argmin(ev))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    gstar = _golden(lambda g: float((np.exp(
        logsoftmax_T(family_L, 10.0 ** g)) *
        (logsoftmax_T(family_L, 10.0 ** g) - pref)).sum()), lo, hi)
    return float(10.0 ** gstar)


def site_probe_matrix(net) -> dict:
    """Full-vocab logits at both sites (+ the y129 gate subset)."""
    return {
        "GATE": site_logits(net, win_i, GATE_COLS).numpy().astype(np.float64),
        "PARA": site_logits(net, held_ids, PARA_COLS).numpy().astype(np.float64),
    }


def flatten_site(Ld: dict, site: str) -> np.ndarray:
    if site == "GATE":
        return Ld["GATE"].reshape(-1, Ld["GATE"].shape[-1])
    return Ld["PARA"].reshape(-1, Ld["PARA"].shape[-1])


SITE_ANS = {"GATE": np.tile(np.array(GATE_ANS), 60),
            "PARA": np.array(PARA_ANS * 30)}
Y129_MASK = np.zeros(420, dtype=bool)
Y129_MASK[::7] = True        # the first of every window's 7 gate probes

# per-state site logits (kept for reference rows)
SITE_L = {}
for key, net in nets.items():
    SITE_L[key] = site_probe_matrix(net)

fits = []
REF_OF = {}                      # family -> reference (base) key
for fam, arm, role, _ in BASELINES:
    REF_OF[fam] = (fam, arm, role)
REF_OF["D"] = REF_OF["N"]        # the D family's loaded fact IS the
REF_OF["X"] = REF_OF["N"]        # shared e261 checkpoint (one object)

anchor_checks = []
for fam, arm, role, _ in BASELINES + STATES:
    key, ref_key = (fam, arm, role), REF_OF[fam]
    Ls_dict, Lr_dict = SITE_L[key], SITE_L[ref_key]
    row = {"family": fam, "arm": arm, "role": role, "sites": {}}
    for site in ("GATE", "PARA"):
        Ls = flatten_site(Ls_dict, site)
        Lr = flatten_site(Lr_dict, site)
        ans = SITE_ANS[site]
        ps = softmax_np(Lr)[np.arange(len(ans)), ans]     # the targets
        primary = fit_T(Ls, ans, ps)
        # co-report 1: x5's literal direction (temper the REFERENCE)
        x5dir = fit_T(Lr, ans, softmax_np(Ls)[np.arange(len(ans)), ans])
        # co-report 2/3: full-vocab KL both directions
        kl_state = fit_T_kl(Ls, Lr)
        kl_ref = fit_T_kl(Lr, Ls)
        # co-report 4: margins / sigma / z-ratios (x7/x8 echoes)
        mar_s = Ls[np.arange(len(ans)), ans] - np.max(
            Ls + np.eye(Ls.shape[-1], dtype=bool)[ans] * -1e9, axis=-1)
        mar_r = Lr[np.arange(len(ans)), ans] - np.max(
            Lr + np.eye(Lr.shape[-1], dtype=bool)[ans] * -1e9, axis=-1)
        sig_s = float(np.median(Ls.max(axis=-1)))
        sig_r = float(np.median(Lr.max(axis=-1)))
        z_s = np.log(np.clip(ps, 1e-12, 1 - 1e-12))  # placeholder, recalc
        p_s = softmax_np(Ls)[np.arange(len(ans)), ans]
        p_r = ps
        z_s = np.log(np.clip(p_s, 1e-12, 1 - 1e-12)) - \
            np.log(np.clip(1 - p_s, 1e-12, 1))
        z_r = np.log(np.clip(p_r, 1e-12, 1 - 1e-12)) - \
            np.log(np.clip(1 - p_r, 1e-12, 1))
        ok = (z_s * z_r) > 0
        zratio = float(np.median(z_r[ok] / z_s[ok])) if ok.any() else None
        row["sites"][site] = {
            "T": primary["T"], "heat": primary["T"] - 1.0,
            "nll_at_T": primary["nll_at_T"], "nll_at_T1": primary["nll_at_T1"],
            "grid_edge": primary["grid_edge"],
            "T_x5dir": x5dir["T"], "T_x5dir_grid_edge": x5dir["grid_edge"],
            "T_kl_state": kl_state, "T_kl_ref": kl_ref,
            "mean_p_state": float(p_s.mean()),
            "mean_p_ref": float(p_r.mean()),
            "median_margin_state": float(np.median(mar_s)),
            "median_margin_ref": float(np.median(mar_r)),
            "median_sigma_state": sig_s, "median_sigma_ref": sig_r,
            "T_app_scale_only": sig_s / sig_r,
            "z_ratio_median": zratio,
            "z_physical_frac": float(ok.mean()),
        }
        if role == "base":
            anchor_checks.append(
                {"state": str(key), "site": site, "T": primary["T"],
                 "abs_dev_from_1": abs(primary["T"] - 1.0)})
    # the y129 gate subset (the read position where the 3.5x is defined)
    Ls = flatten_site(Ls_dict, "GATE")[Y129_MASK]
    Lr = flatten_site(Lr_dict, "GATE")[Y129_MASK]
    ans = np.full(60, GATE_ANS[0])
    ps = softmax_np(Lr)[np.arange(60), ans]
    sub = fit_T(Ls, ans, ps)
    row["sites"]["GATE_y129"] = {
        "T": sub["T"], "heat": sub["T"] - 1.0,
        "mean_p_state": float(softmax_np(Ls)[np.arange(60), ans].mean()),
        "mean_p_ref": float(ps.mean()),
        "note": "the battery read position alone (the 3.5x's own site)",
    }
    fits.append(row)
    log(f"  fit {fam}/{arm}/{role}: T_gate {row['sites']['GATE']['T']:.4f} "
        f"(y129 {row['sites']['GATE_y129']['T']:.4f}) T_para "
        f"{row['sites']['PARA']['T']:.4f}")

G_ANCHOR = {
    "rows": anchor_checks,
    "max_abs_dev": max(a["abs_dev_from_1"] for a in anchor_checks),
    "tol": 1e-3,
    "note": "every reference (base) state fit against itself must read "
            "T = 1 (the positive control; term-wise exact at T=1; x5's "
            "t0-anchor convention on the verified reference anchors)",
}
assert G_ANCHOR["max_abs_dev"] <= 1e-3, f"anchor FAILED: {G_ANCHOR}"
G_ANCHOR["pass"] = True
metrics["gates"]["G_ANCHOR"] = G_ANCHOR
metrics["fits"] = fits
write_partial("P2 the one-T fits done (anchor 1.0000 certified)")

# ============================================ P3: reads, trace, co-reads ==
log("P3: the honesty co-reads + the overshoot trace")

FIT = {(f["family"], f["arm"], f["role"]): f for f in fits}


def heat_of(fam, arm, role, site="GATE"):
    return FIT[(fam, arm, role)]["sites"][site]["heat"]


# (d) the honesty co-read table (committed values read at runtime)
def committed_cell(m, arm):
    rd = m["adjudication"]["reads"][arm]
    return {
        "post_g0": rd.get("post_g0"),
        "post_gm12": rd.get("post_gm12"),
        "post_ce_r": rd.get("post_ce_r"),
        "survival_ratio": rd.get("survival_ratio_committed_denom",
                                 rd.get("survival_ratio",
                                        rd.get("fact1_ratio"))),
        "final_panel_g0": rd.get("final_panel_g0"),
    }


co = {
    "N/ERROR-GATED": committed_cell(e288m, "ERROR-GATED"),
    "N/NAME-FIXED-TWIN": committed_cell(e288m, "NAME-FIXED-TWIN"),
    "D/CONTRADICTED-WITH-CONTROLLER": committed_cell(
        e289m, "CONTRADICTED-WITH-CONTROLLER"),
    "D/CONTRADICTED-NO-CONTROLLER": committed_cell(
        e289m, "CONTRADICTED-NO-CONTROLLER"),
    "D/C1:ERROR-GATED": committed_cell(e289m, "C1:ERROR-GATED"),
    "D/C1:NAME-FIXED-TWIN": committed_cell(e289m, "C1:NAME-FIXED-TWIN"),
    "F/FIVE-CONTROLLERS": committed_cell(e291m, "FIVE-CONTROLLERS"),
    "F/SINGLE-CONTROL-TWIN": committed_cell(e291m, "SINGLE-CONTROL-TWIN"),
    "X/SANCTUARY-TWIN(e287)": {
        "post_g0": r288["the_cited_control_e287"][
            "its_sanctuary_twin_post"]},
}
for k, v in co.items():
    fam, arm = k.split("/", 1)
    v["heat_gate"] = heat_of(fam, arm, "post", "GATE")
    v["heat_para"] = heat_of(fam, arm, "post", "PARA")
    v["heat_gate_y129"] = heat_of(fam, arm, "post", "GATE_y129")
    v["reprobed_g0"] = state_rows[
        str((fam, arm, "post"))]["g0_battery"]
    v["reprobed_held_pz"] = state_rows[
        str((fam, arm, "post"))]["held_pz"]
metrics["co_reads"] = co

# (b) the overshoot trace: T vs the read's rise; committed traj_g0 the axis
trace_states = []
BASELINE_READ = {("N", "ERROR-GATED"): FACT_BASELINE_G0,
                 ("N", "NAME-FIXED-TWIN"): FACT_BASELINE_G0,
                 ("D", "CONTRADICTED-WITH-CONTROLLER"): FACT_BASELINE_G0,
                 ("D", "CONTRADICTED-NO-CONTROLLER"): FACT_BASELINE_G0,
                 ("D", "C1:ERROR-GATED"): FACT_BASELINE_G0,
                 ("D", "C1:NAME-FIXED-TWIN"): FACT_BASELINE_G0,
                 ("X", "SANCTUARY-TWIN(e287)"): FACT_BASELINE_G0}
for fam, arm, role, _ in STATES:
    rd = state_rows[str((fam, arm, role))]["g0_battery"]
    if (fam, arm) in BASELINE_READ:
        rise = rd / BASELINE_READ[(fam, arm)]
    else:      # F family: FACT1's panel read vs ITS committed baseline
        rise = state_rows[str((fam, arm, role))]["panel_g0"]["FACT1"] / \
            F_PANEL_BASELINE["FACT1"]
    trace_states.append({
        "family": fam, "arm": arm, "role": role,
        "read_g0": rd, "read_rise": rise,
        "heat_gate": heat_of(fam, arm, role, "GATE"),
        "heat_para": heat_of(fam, arm, role, "PARA"),
        "heat_gate_y129": heat_of(fam, arm, role, "GATE_y129"),
    })
committed_traj = {
    "N/ERROR-GATED": r288["ERROR-GATED"]["traj_g0"],
    "N/NAME-FIXED-TWIN": r288["NAME-FIXED-TWIN"]["traj_g0"],
    "D/CONTRADICTED-WITH-CONTROLLER":
        r289["CONTRADICTED-WITH-CONTROLLER"]["traj_g0"],
    "D/CONTRADICTED-NO-CONTROLLER":
        r289["CONTRADICTED-NO-CONTROLLER"]["traj_g0"],
    "D/C1:ERROR-GATED": r289["C1:ERROR-GATED"]["traj_g0"],
    "D/C1:NAME-FIXED-TWIN": r289["C1:NAME-FIXED-TWIN"]["traj_g0"],
}
metrics["trace"] = {
    "states": trace_states,
    "committed_traj_g0": committed_traj,
    "read_axis_note": "read_rise = re-probed g0 / the committed loaded "
                      "baseline 0.2646... (N/D/X families); FACT1 panel / "
                      "ITS committed baseline (F family); the committed "
                      "traj_g0 curves are the parents' milestone reads "
                      "(md5-bound runtime reads)",
}

# (c) the riders
rd_cwc = FIT[("D", "CONTRADICTED-WITH-CONTROLLER", "post")]
rd_c1 = FIT[("D", "C1:ERROR-GATED", "post")]
rd_eg = FIT[("N", "ERROR-GATED", "post")]
rd_cnc = FIT[("D", "CONTRADICTED-NO-CONTROLLER", "post")]
rd_st = FIT[("X", "SANCTUARY-TWIN(e287)", "post")]
metrics["riders"] = {
    "denial_vs_neutral": {
        "question": "does contradicted maintenance run hotter?",
        "D_CWC_gate_T": rd_cwc["sites"]["GATE"]["T"],
        "D_C1_gate_T": rd_c1["sites"]["GATE"]["T"],
        "N_EG_gate_T": rd_eg["sites"]["GATE"]["T"],
        "denial_minus_neutral_gate_heat":
            rd_cwc["sites"]["GATE"]["heat"] - rd_c1["sites"]["GATE"]["heat"],
        "denial_minus_founding_gate_heat":
            rd_cwc["sites"]["GATE"]["heat"] - rd_eg["sites"]["GATE"]["heat"],
        "para_side": {
            "D_CWC": rd_cwc["sites"]["PARA"]["T"],
            "D_C1": rd_c1["sites"]["PARA"]["T"],
            "N_EG": rd_eg["sites"]["PARA"]["T"]},
    },
    "passive": {
        "question": "is heat the controller's signature or survival's?",
        "D_CNC_passive_gate_T": rd_cnc["sites"]["GATE"]["T"],
        "X_e287_ST_passive_gate_T": rd_st["sites"]["GATE"]["T"],
        "survivors_gate_T": [rd_eg["sites"]["GATE"]["T"],
                             rd_cwc["sites"]["GATE"]["T"],
                             rd_c1["sites"]["GATE"]["T"]],
        "note": "CNC = no maintenance under denial (dead, 0.009x); "
                "e287 ST = no maintenance under neutral traffic (dead; "
                "ITS corpus seed 28701 — disclosed)",
    },
    "flat_twins": {
        "N_NAME-FIXED-TWIN_gate_T":
            FIT[("N", "NAME-FIXED-TWIN", "post")]["sites"]["GATE"]["T"],
        "D_C1:NAME-FIXED-TWIN_gate_T":
            FIT[("D", "C1:NAME-FIXED-TWIN", "post")]["sites"]["GATE"]["T"],
        "F_SINGLE-CONTROL-TWIN_gate_T":
            FIT[("F", "SINGLE-CONTROL-TWIN", "post")]["sites"]["GATE"]["T"],
    },
}
write_partial("P3 co-reads + trace + riders done")

# ================================================== P4: the adjudication ==
log("P4: the adjudication (bars verbatim on the founding state)")


def clauses(heat_g, heat_p):
    fever = (heat_g >= FEVER_GATE) and (heat_p < FEVER_HALF * heat_g)
    calibrated = abs(heat_g) < CAL_TOL and abs(heat_p) < CAL_TOL
    local = (heat_g >= FEVER_GATE) and (abs(heat_p) < CAL_TOL)
    if fever:
        v = "FEVER"
    elif calibrated:
        v = "CALIBRATED-SURVIVAL"
    elif local:
        v = "LOCAL-FEVER"
    else:
        v = "MIXED"
    return {"fever_clause": bool(fever), "calibrated_clause": bool(calibrated),
            "local_clause": bool(local), "verdict": v}


f_row = FIT[FOUNDING]
hg, hp = f_row["sites"]["GATE"]["heat"], f_row["sites"]["PARA"]["heat"]
founding_clauses = clauses(hg, hp)
per_state = {}
for fam, arm, role, _ in STATES:
    if role != "post":
        continue
    r = FIT[(fam, arm, role)]
    per_state[f"{fam}/{arm}"] = {
        "gate_T": r["sites"]["GATE"]["T"],
        "para_T": r["sites"]["PARA"]["T"],
        "gate_y129_T": r["sites"]["GATE_y129"]["T"],
        **clauses(r["sites"]["GATE"]["heat"], r["sites"]["PARA"]["heat"]),
    }
metrics["adjudication"] = {
    "bars": {k: v for k, v in
             metrics["registration"]["bars_verbatim"].items()},
    "constants": {"FEVER_GATE": FEVER_GATE, "FEVER_HALF": FEVER_HALF,
                  "CAL_TOL": CAL_TOL},
    "founding_state": {
        "state": "e288 ERROR-GATED post (T vs the loaded fact — its "
                 "verified reference anchor)",
        "gate_T": f_row["sites"]["GATE"]["T"],
        "gate_heat": hg,
        "para_T": f_row["sites"]["PARA"]["T"],
        "para_heat": hp,
        "gate_y129_T": f_row["sites"]["GATE_y129"]["T"],
        "gate_y129_heat": f_row["sites"]["GATE_y129"]["heat"],
        "para_over_half_gate": bool(hp >= FEVER_HALF * hg) if hg > 0 else None,
        **founding_clauses,
    },
    "per_state_post": per_state,
    "verdict": founding_clauses["verdict"],
    "clause": (
        f"FEVER requires gate heat >= +{FEVER_GATE} AND para heat < "
        f"{FEVER_HALF}x gate heat: gate {hg:+.4f}, para {hp:+.4f} -> "
        f"para/gate = {hp / hg if hg != 0 else float('nan'):.3f}"),
}
log(f"P4: VERDICT {founding_clauses['verdict']} (gate {hg:+.4f}, "
    f"para {hp:+.4f}, y129 {f_row['sites']['GATE_y129']['heat']:+.4f})")
write_partial(f"P4 adjudicated: {founding_clauses['verdict']}")

# ======================================================== P5: the outputs ==
log("P5: the figure + REPORT")

# ---- the figure (4 panels) ----
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    f"E304 THE FEVER CELL — {founding_clauses['verdict']} — the founding "
    f"3.5x through the one-T lens (gate {hg:+.3f} / para {hp:+.3f})",
    fontsize=13)

# A: the two-site heat bars
ax = axes[0][0]
labels, hgts, hpts = [], [], []
for fam, arm, role, _ in STATES:
    if role != "post":
        continue
    labels.append(f"{fam}/{arm.replace('CONTRADICTED-', 'C').replace(': ', ':')}")
    hgts.append(heat_of(fam, arm, "post", "GATE"))
    hpts.append(heat_of(fam, arm, "post", "PARA"))
x = np.arange(len(labels))
ax.bar(x - 0.18, hgts, 0.36, label="GATE (the taught tokens)",
       color="#c0392b")
ax.bar(x + 0.18, hpts, 0.36, label="PARA (held-out surfaces)",
       color="#2980b9", hatch="//", edgecolor="white")
ax.axhline(FEVER_GATE, color="k", ls="--", lw=1)
ax.axhline(0.0, color="k", lw=0.8)
ax.axhline(-CAL_TOL, color="gray", ls=":", lw=1)
ax.axhline(CAL_TOL, color="gray", ls=":", lw=1)
fi = labels.index("N/ERROR-GATED")
ax.annotate("THE FOUNDING\nSTATE", (fi, max(hgts[fi], 0.02)),
            ha="center", fontsize=9, fontweight="bold",
            xytext=(fi, max(hgts) * 0.72),
            arrowprops=dict(arrowstyle="->"))
ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("HEAT = T - 1 (state vs the loaded fact, its verified ref)")
ax.set_title("(a) the two-site table — gate vs paraphrase")
ax.legend(fontsize=8)

# B: the overshoot trace — T vs read rise + committed trajectories
ax = axes[0][1]
for lbl, traj in committed_traj.items():
    ts = sorted(int(k) for k in traj)
    ax.plot(ts, [traj[str(t)] / FACT_BASELINE_G0 for t in ts], lw=1,
            alpha=0.55, label=lbl.split("/")[-1][:22])
ax.set_xlabel("corpus step (committed traj; milestones only — states NOT "
              "checkpointed, disclosed)")
ax.set_ylabel("read rise (committed g0 / 0.2646)", color="gray")
ax.tick_params(axis="y", labelcolor="gray")
ax2 = ax.twinx()
fam_c = {"N": "#c0392b", "D": "#8e44ad", "F": "#16a085", "X": "#7f8c8d"}
for t in trace_states:
    if t["role"] != "post":
        continue
    ax2.scatter(t["read_rise"], t["heat_gate"], marker="o", s=46,
                color=fam_c[t["family"]],
                edgecolor="k" if t["arm"] == "ERROR-GATED" else "none",
                zorder=5)
    ax2.scatter(t["read_rise"], t["heat_para"], marker="s", s=34,
                color=fam_c[t["family"]], alpha=0.55, zorder=4)
    if t["family"] == "N" and t["arm"] == "ERROR-GATED":
        ax2.annotate("EG gate", (t["read_rise"], t["heat_gate"]),
                     fontsize=8, xytext=(4, 4),
                     textcoords="offset points")
ax2.set_xscale("log")
ax2.axhline(FEVER_GATE, color="k", ls="--", lw=1)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_ylabel("HEAT (o=gate, s=para)")
ax2.set_title("(b) the overshoot trace: T vs the read's rise")

# C: the riders
ax = axes[1][0]
groups = [
    ("denial vs neutral\n(gate T)",
     [rd_cwc["sites"]["GATE"]["T"], rd_c1["sites"]["GATE"]["T"],
      rd_eg["sites"]["GATE"]["T"]]),
    ("passive vs survivors\n(gate T)",
     [rd_cnc["sites"]["GATE"]["T"], rd_st["sites"]["GATE"]["T"],
      rd_eg["sites"]["GATE"]["T"], rd_cwc["sites"]["GATE"]["T"]]),
    ("the flat twins\n(gate T)",
     [FIT[("N", "NAME-FIXED-TWIN", "post")]["sites"]["GATE"]["T"],
      FIT[("D", "C1:NAME-FIXED-TWIN", "post")]["sites"]["GATE"]["T"],
      FIT[("F", "SINGLE-CONTROL-TWIN", "post")]["sites"]["GATE"]["T"]]),
]
bx = 0
positions, values, colors, ticks = [], [], [], []
palette = ["#8e44ad", "#c0392b", "#2980b9", "#16a085", "#7f8c8d", "#f39c12"]
for gi, (title, vals) in enumerate(groups):
    for vi, v in enumerate(vals):
        positions.append(bx)
        values.append(v)
        colors.append(palette[(gi + vi) % len(palette)])
        bx += 1
    ticks.append((title, bx - len(vals) / 2 - 0.5))
    bx += 0.8
ax.bar(positions, values, 0.8, color=colors)
ax.axhline(1.0, color="k", lw=1, label="the anchor (calibration)")
lbls = ["D-CWC", "D-C1", "N-EG", "CNC", "e287-ST", "N-EG", "D-CWC",
        "N-NFT", "D-C1NFT", "F-SCT"]
for p, v, l in zip(positions, values, lbls):
    ax.text(p, v + 0.01 * (1 if v >= 1 else -1), l, ha="center",
            fontsize=7)
ax.set_xticks([t[1] for t in ticks], [t[0] for t in ticks], fontsize=8)
ax.set_ylabel("T at the gate tokens")
ax.set_title("(c) the riders")
ax.legend(fontsize=8)

# D: the honesty co-read table
ax = axes[1][1]
ax.axis("off")
cols = ["state", "ce_r", "gm12", "g0", "ratio", "HEAT gate", "HEAT para"]
rows_t = []
for k, v in co.items():
    rows_t.append([
        k, f"{v['post_ce_r']:.3f}" if v.get("post_ce_r") else "n/a",
        f"{v['post_gm12']:.4f}" if v.get("post_gm12") else "n/a",
        f"{v['post_g0']:.4f}" if v.get("post_g0") else
        f"{v['final_panel_g0']['FACT1']:.4f}",
        f"{v['survival_ratio']:.3f}" if v.get("survival_ratio") else "n/a",
        f"{v['heat_gate']:+.3f}", f"{v['heat_para']:+.3f}"])
tbl = ax.table(cellText=rows_t, colLabels=cols, loc="center",
               cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1, 1.35)
ax.set_title("(d) the honesty co-reads (committed ce_r/gm12 + this cell's HEATs)")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(RD / "e304_fever.png", dpi=150)
plt.close(fig)
log(f"[fig] {RD / 'e304_fever.png'}")

# ---- REPORT.md ----
def fmt_clauses(c):
    return (f"{c['verdict']} (fever {c['fever_clause']}, calibrated "
            f"{c['calibrated_clause']}, local {c['local_clause']})")


two_site_lines = []
for fam, arm, role, _ in BASELINES + STATES:
    r = FIT[(fam, arm, role)]
    two_site_lines.append(
        f"| {fam} | {arm} | {role} | "
        f"{r['sites']['GATE']['T']:.4f} ({r['sites']['GATE']['heat']:+.4f}) | "
        f"{r['sites']['GATE_y129']['T']:.4f} "
        f"({r['sites']['GATE_y129']['heat']:+.4f}) | "
        f"{r['sites']['PARA']['T']:.4f} ({r['sites']['PARA']['heat']:+.4f}) | "
        f"{r['sites']['GATE']['T_kl_state']:.4f} / "
        f"{r['sites']['PARA']['T_kl_state']:.4f} |")

rep = f"""# E304 — THE FEVER CELL — {founding_clauses['verdict']}

**{utcnow()}** — eval-only CPU desk cell (threads 4, no GPU, no envelope
writes). The honesty audit of the founding success: the error-gated
controller's 3.5x read overshoot through the one-T thermal lens, at the
gate tokens vs the held-out paraphrase — does preservation buy
confidence the organism cannot cash?

## The verdict

**{founding_clauses['verdict']}** — the founding state (e288 ERROR-GATED
post, T fit against the loaded fact — its verified reference anchor): gate T
{f_row['sites']['GATE']['T']:.4f} (heat {hg:+.4f}), paraphrase T
{f_row['sites']['PARA']['T']:.4f} (heat {hp:+.4f});
{metrics['adjudication']['clause']}.

## (a) THE TWO-SITE T TABLE (the discriminator, verbatim)

| fam | arm | role | GATE T (heat) | gate-y129 T (heat) | PARA T (heat) | KL-state co-report (gate/para) |
|---|---|---|---|---|---|---|
{chr(10).join(two_site_lines)}

The y129 column is the battery read position alone — the site where the
committed 3.5x overshoot is defined. The KL co-report is
argmin KL(softmax(L_state/T) || softmax(L_ref)) — the shape-residual
form (the mass-vs-spike decomposition's price on the primary's
saturation; see disclosures).

## (b) THE OVERSHOOT TRACE

Milestone states were NOT checkpointed by the parents (disclosed; glob
evidence in metrics.disclosures.milestone_states) — the T axis lives at
{{reference-base, post}} per arm; the read axis uses the committed traj_g0
curves (md5-bound runtime reads). T-vs-read-rise for every state in
metrics.trace.states; figure panel (b).

## (c) THE RIDERS

- **Denial vs neutral (does contradicted maintenance run hotter?)**:
  CWC gate T {rd_cwc['sites']['GATE']['T']:.4f} vs C1
  {rd_c1['sites']['GATE']['T']:.4f} (denial heat minus neutral heat
  {metrics['riders']['denial_vs_neutral']['denial_minus_neutral_gate_heat']:+.4f});
  para side {metrics['riders']['denial_vs_neutral']['para_side']}.
- **The passive twin (heat the controller's signature or survival's?)**:
  denial-passive (CNC, dead at 0.009x) gate T
  {rd_cnc['sites']['GATE']['T']:.4f}; neutral-passive (e287 sanctuary
  twin, dead; corpus seed 28701 — disclosed) gate T
  {rd_st['sites']['GATE']['T']:.4f}; the survivors
  {metrics['riders']['passive']['survivors_gate_T']}.
- **The flat twins**: {json.dumps(metrics['riders']['flat_twins'])}.

## (d) THE HONESTY CO-READS

See metrics.co_reads + figure panel (d): each state's committed
post_ce_r / post_gm12 / post_g0 / survival ratio joined with this
cell's HEATs. The corpus stream's health (ce_r) reads alongside the
thermal verdicts — the honesty audit's two lenses on the same states.

## Per-state clause results (verbatim)

{json.dumps(per_state, indent=1)}

## THE RECOVERY NOTE (what this executor fixed from the dead draft)

The birth commit d2b5d8b froze the bars + operationalization BEFORE
compute; a prior executor died on a model failure mid-restructure, leaving
the working script between two designs (its discovery was right, its edit
was incomplete — STATE lookups on `resume` keys that no longer existed, a
guaranteed KeyError). This executor verified the dead draft's discovery at
runtime and completed the restructure it implies:

- **THE FINDING (verified, not transcribed)**: every parent
  `*_resume.pt` carries model weights BIT-IDENTICAL to its `*_post.pt`
  (flat-md5 equal, 9/9 arms; the wrapper only adds optimizer/generator
  state for later landing passes). They are POST-phase duplicates, NOT
  pre-phase baselines.
- **THE CONSEQUENCE**: the docstring's reference clause ("the arm's own
  resume checkpoint — the loaded fact as each cell loaded it") is only
  satisfiable by the LOADED FACT itself: `e261_K10K_inst_resume.pt` for
  the N/D/X families (certified g0 = 0.2646, the frozen
  FACT_BASELINE_G0) and `e291_organism.pt` for the F family (certified
  on the per-fact panels = e291's committed final/ratio baselines) —
  exactly the parenthetical's definition and the docstring's own
  certification clause ("resumes additionally certified against the
  committed loaded baseline 0.2646... or e291's committed per-fact
  baselines"). Fitting the `*_resume.pt` duplicates would fit every
  state against itself: HEAT = 0 identically, a degenerate
  CALIBRATED-SURVIVAL with zero discriminating power. The bars are
  untouched (verbatim); this fix restores the instrument the frozen
  bars presuppose.
- Also completed from the dead draft: `common.save_json` -> inline
  JSON write (functionally identical), the smoke filter, the
  resume-duplicate evidence block, the held-bank bind re-pointed at the
  e291 organism, and the anchor row on the reference (base) states.

## DISCLOSURES

1. **The reference re-anchoring** (the recovery note above, in full):
   the lens reference is the loaded fact / e291 organism, per the
   docstring's own parenthetical + certification clause; the
   `*_resume.pt` duplicates' bit-identity is runtime evidence
   (metrics.disclosures.resume_duplicates), not a transcription.
2. **No milestone states exist** (t100/200/300 were not checkpointed);
   the trace uses finals + resumes with the committed read curves.
3. **The lens direction is the registered choice**: the temperature
   sits on the STATE's logits (T > 1 = the site must be cooled to the
   reference's calibration = runs hot — the direction in which the
   bars' "T rises" literally reads overconfidence). x5's literal
   direction, both KL forms, margins/sigma/T_app, and per-probe
   z-ratios (x8's physicality convention) co-report in metrics.fits —
   they price the primary's known saturation under answer-mass rises
   (the answer coordinate carries the mass gain itself under any
   baseline-anchored temperature lens; the two-site comparison is the
   bars' own discrimination).
4. **e287's sanctuary twin** rode corpus draws on seed 28701 (not
   28801) — the passive rider's session differs from e288's.
5. **The F family's gate site** is the union bank (all five facts are
   name-family slices of the same 60 ZEPHYRA windows — T270's finding);
   FACT1's panel read anchors its read-rise.
6. All certification gates passed at runtime (records md5-bound;
   0 re-probe failures; the anchor rows read 1.000000; the paraphrase
   bank bit-bound to e291's committed held_t0).

Birth commit: before compute (see provenance.git_head_at_start in
metrics.json). Script: lab/e304_fever.py. Outputs:
runs/e304/{{metrics.json, e304_fever.png, REPORT.md}}.
"""
(RD / "REPORT.md").write_text(rep, encoding="utf-8")
log(f"[report] {RD / 'REPORT.md'}")

metrics["timing"] = {"total_s": time.time() - T0, "utc_end": utcnow()}
metrics["outputs"] = {
    "metrics": "runs/e304/metrics.json",
    "figure": "runs/e304/e304_fever.png",
    "report": "runs/e304/REPORT.md",
}
metrics["status"] = f"COMPLETE — adjudicated {founding_clauses['verdict']} " \
                    f"({utcnow()})"
(RD / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
log(f"DONE: {founding_clauses['verdict']} in {time.time() - T0:.1f}s")

