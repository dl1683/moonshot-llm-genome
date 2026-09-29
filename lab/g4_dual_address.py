"""G4 — THE DUAL-ADDRESS NET (W020 generative turn; design + registration:
scratch/g4_design.md, COMMITTED 2026-09-29 BEFORE this implementation — the
bars below are VERBATIM from that document; adjudicate against exactly this,
no bar shopping).

THE QUESTION: are THE ERROR COMPASS and THE VARIANCE SWITCH architectural
necessities or pre-LN-transformer contingencies? The lab's every dissection
confounded "address" with "position" (the address WAS a wpe row). g4
deconfounds them: DualGPT = the lab TinyGPT (4L/4H/128d/256ctx, 840,704
params) with the input floor split into TWO address tables — `wpe` (the P
floor, position-only; attribute name kept so every state-dict wpe instrument
runs VERBATIM) plus a K=32-slot content-conditional codebook (the A floor)
selected per-token by a small gate on [wte(x_t); wpe(t)] — 863,328 params.
The gate sees exactly ONE token (the representational asymmetry that is the
experiment's engine). A matched single-table control root (same size, same
seed, same recipe) runs the identical program FIRST (the scale gate).

PROGRAM: stage 0 pretrain both roots -> THE SPINE (gate PSI + slot-usage
mass, measured BEFORE any teaching; its registered table predicts the
carrier) -> stage 1 e043 install -> stage 2 the e147 minimal ladder
{locked, +-1, (+-4), +-8} -> stage 3 the e143 NEAR/FAR compass arms ->
stage 4 the e176N neutral wash -> stage 5 the deletion hierarchy
(D-P-site, D-P-0, D-A-top3, D-A-all, GATE-FREEZE, the e160 N2 knife).

REGISTERED PREDICTIONS (spec sec 6, VERBATIM bars; frozen per-law):
  P0 SPINE: the gate's pre-teaching class (spec sec 4 table) predicts the
     install carrier AND the w>=1 carrier. FIRES on census-match; DIES on
     contradiction; mixed/borderline => SPINE-UNSHARP (texture, no bars).
  P1 COMPASS (committed: COMPASS-IS-POSITIONAL): NEAR builds P-site content
     at rows 5-13 (e116 criterion AND strength >= 2x shared-control max),
     FAR at 137-143; A-slots carry no fact content (A-arm strength < 0.5 x
     P-site strength); row-0 presence at/below install baseline; NEAR
     novel-geometry ~0 (site-bound). Falsifiers: COMPASS-CONTENT (NEAR rides
     A-slots, A-arm >= 2x P-site); COMPASS-DEAD (NEAR ~= FAR on every floor).
  P2 SWITCH (committed: CLIFF-ON-P, FLIGHT-TO-HEADS): on BOTH roots
     A_P(0) >= +0.15; A_P(1) <= 0 (w* = 1); A_A(w) fact-free at every w
     (strength < 0.10); NR onsets at w=1 (>= 2x root); head top-2 share
     rises w0 -> w8. Falsifiers: (a) SLOT-SUCCESSION (any w >= 1: A_A >=
     0.15 at CE <= +0.35); (b) NO-SWITCH (A_P flat max-min <= 0.10, or
     w0/w1 same carrier ON THE CONTROL); (c) NO-SWITCH on dual while
     control cliffs. Order: committed -> (a) -> (c) -> (b).
  P3 BASIN (committed: DISSOLVES-BY-TWO-STEPS, BOTH roots, ANY carrier):
     g-12 p(Z) crosses < 0.05 within (1, 2] steps of the e176N neutral
     stream at lr 1e-3. Falsifier: ANY-SURVIVOR (g-12 >= 0.5 at +50 on
     either root) — W019's standing debt, paid here for free.
  P4 SURGERY (committed: THE HIERARCHY SEGREGATES BY FLOOR): (a) locked/
     P-carried fact dies under D-P-site (>= 60% drop at CE <= +0.35);
     (b) jitter fact survives every TABLE surgery (D-P-site, D-A-top3,
     D-A-all: < 30% drop at any CE) and dies under the N2-class head
     ablation (>= 60% at CE <= +0.35); (c) GATE-FREEZE spares the fact
     whenever the carrier is P or heads (< 10% drop); (d) D-A-all costs
     the organism (+0.2..+1.0 CE) but kills no fact.
  GLOBAL VERDICT TABLE (spec sec 7): control cliff? x dual outcome =>
     NECESSITY / LAWS-REAL-CARRIER-CONTINGENT / INVENTORY-SENSITIVE /
     SCALE-LINEAGE-BOUND.

OPERATIONALIZATIONS (frozen here before compute):
  * A_P(w) = e140 census VERBATIM on wpe: row-129 mean/zero replacement
    drop on the install-60 g0 battery (ids130), strength = min(mean-drop,
    zero-drop). A_A(w) = the same census ported to slots: the arm's top-3
    READ-BAND (cols 120-140) slots by usage mass, replaced jointly by
    mean-of-slots / by 0, drop on ids130, strength = min.
  * NR(w) = e141 d_r0: zero-arm row-0 drop at g-12 (primary) and g+12
    (robustness) on the install-60 battery; root baseline = the INSTALLED
    root; onset = NR >= 2x root. Co-report: D-A-top at g-12 (the slot-sink
    dial) and the brake = p(D-P-site, g0) - p(none, g0).
  * carrier(w): P if A_P >= 0.15 and A_A < 0.15; A if A_A >= 0.15 and
    A_P < 0.15; HEADS if g-12 expression >= 0.30 and both tables < 0.15;
    else UNRESOLVED (numbers reported). SLOT-SUCCESSION additionally
    requires the top3-slot zero-cell CE <= +0.35.
  * P1 site_pos = e116 criterion AND strength >= 2x shared-control max
    (rows {60,100,150,160,170,200,220}, e143 convention); census readout =
    the arm's OWN trained geometry (e139 adaptation); A-arm census = the
    arm's top-3 read-row slots, readout = own-geometry onset; row-0
    at/below baseline = e143's midpoint bar with in-run references
    (install root baseline; dual w8 arm as the routed reference); novel
    ~0 = mean over g{-12,-2,+2,+12} <= 0.10 (family-1 NEAR read 0.0018,
    FAR 0.205 — the bar separates them).
  * P4 %drop = 1 - p_cell/p_none at the net's primary geometry (locked:
    g0; jitter: g-12, the field read; both geometries reported for every
    cell); flat-CE bar CE_cell - CE_arm <= +0.35. N2-class knife = e160
    escalation: per-head census -> top-4 singles -> N2 pair -> (N3, E4
    escalation only while the bar is unmet).
  * SPINE table (spec sec 4, verbatim thresholds): PSI_read < 0.25 AND
    M(top read slot) > 2% => locked P / w>=1 HEADS; PSI_read < 0.25 AND
    M < 0.2% => A / A; PSI_read > 1 => P / HEADS; else SPINE-UNSHARP.
    PSI = median KL(a(x,t)||a(x,t+1)) / median KL(a(x,t)||a(x',t)) over
    4,000 corpus positions (2,000 in the read band cols 120-140); M(k) =
    argmax-slot usage mass.

INSTRUMENT PROVENANCE: battery_cell / battery_pz / ce_fixed_cpu /
val_windows / deleted_wpe / read_fact_at / row_census / finetune_arm are
lab/e143_error_steering.py's copies VERBATIM (the e131/e119/e113/e068/e065/
e043 lineage; finetune_arm = e109 arm-b recipe with the bounded e119
gate_launch and DEV parameterized) — model-agnostic so DualGPT runs the
identical program. The neutral wash is lab/e176n_neutral_wash.py's
finetune_freeze arithmetic (e176 verbatim) with e170's neutral anchor bank.
The head census/knife is common.lesion's c_proj pre-hook convention
(e001/e038/e133/e160 lineage). NEW instrument code (~the only new readers):
DualGPT, the gate table/PSI probe, slot-usage mass, slot census, fact-read
slot selection, GATE-FREEZE swap, deleted_slots.

COMPUTE ENVELOPE: GPU lane (gpu_ok() double-poll per launch, NO concurrent,
cooldown(60) between trainings, per-training cap 1800 s; pretrains chunked
<= 3 x 175 s per root per the spec's <= 3 x 180 s budget); every census is
CPU-side on state-dict snapshots (8 threads). SMOKE env G4_SMOKE=1 runs a
reduced shakedown (nothing adjudicated).

Outputs: runs/g4/{metrics.json (written after EVERY stage — fail-late
avoided), ladder.png, compass.png, wash.png, dual_address.png}; checkpoints
runs/checkpoints/g4_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit, no push.

Run:  cd lab && python g4_dual_address.py    (G4_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # GPU lane for trainings

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e143 convention

import torch.nn as nn                                 # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Block, Cfg, CharCorpus, TinyGPT, cooldown,     # noqa: E402
                    gpu_ok, gpu_status, lesion, run_dir, save_json,
                    set_seed, train_model)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("G4_SMOKE") == "1"
CPU = torch.device("cpu")
_USE_GPU = torch.cuda.is_available() and gpu_ok()
DEV = torch.device("cuda") if _USE_GPU else CPU

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

CORPUS_SEED = 1337
NET_SEED = 4305                   # spec: net seed 4305 (both roots)
GATE_RESEED = 4306                # the ONE gate-only re-seed remedy
INSTALL_SEED = 10901              # e113 CONS_SEED convention
ARM_SEED = 10901                  # e147 ladder convention
NEAR_FAR_SEED = 10902             # e143 L_SEED convention
WASH_SEED = 10902                 # e176n FREEZE_SEED convention
E170_ANCHOR_SEED = 170            # e170 neutral bank seed
R_EVAL_SEED = 26502               # e065 CE_R bank seed
PSI_SEED = 43051                  # spine sampling
SLOT_SEED = 43052                 # slot-usage sampling

# ---- architecture -------------------------------------------------------------
N_LAYER, N_HEAD, N_EMBD = 4, 4, 128
N_SLOTS = 32
GATE_HID = 64
BASE_PARAMS = 840_704             # 4L/4H/128d/256 TinyGPT (spec sec 3)
DUAL_PARAMS = 863_328             # + slots 4,096 + gate 18,528 (spec sec 3)

# ---- protocol geometry (e143/e147 verbatim) ------------------------------------
NEAR_PRE = 6                      # ZEPHYRA x-cols 6..12; read rows 5..11
NEAR_ADDR_ROW = NEAR_PRE - 1      # 5
NEAR_Z_XCOL = NEAR_PRE            # 6
NEAR_CONT = BLOCK - NEAR_PRE - len(NAME)          # 243
NEAR_SITE_ROWS = tuple(range(5, 14))
FAR_J = 8                         # e109's +8 offset pool
FAR_ADDR_ROW = PRE - 1 + FAR_J                    # 137
FAR_Z_XCOL = PRE + FAR_J                          # 138
FAR_SITE_ROWS = tuple(range(137, 144))
JITTERS8 = (-8, -4, 0, 4, 8)      # e113 registered jitter set
D_ALL = (121, 125, 129, 133, 137)                 # e113's D-all set
NOVEL_GEO_LADDER = (-12, 12)      # e147: g-12 primary, g+12 robustness
NOVEL_GEO_COMPASS = (-12, -2, 2, 12)              # e143's set
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)   # e143 control band
CONTROL_ROWS = (1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120)  # e140 verbatim
ADDR_BAND = tuple(r for r in range(121, 138))
CENSUS_ROWS = (0,) + CONTROL_ROWS + ADDR_BAND     # e140's row set verbatim
NEAR_ROWS = (0, 1, 2) + tuple(range(3, 18)) + SHARED_CTR
FAR_ROWS = (0, 1, 2) + (134, 135, 136) + tuple(range(137, 147)) + SHARED_CTR
READ_BAND = (120, 140)            # the instruments' read region (spec sec 4)

def offset_grid(w: int) -> tuple[int, ...]:
    """e147's balanced grids: w=1 -> {-1,0,1} (recorded deviation), else
    5-point (-w, -w/2, 0, w/2, w)."""
    if w == 1:
        return (-1, 0, 1)
    h = w // 2
    return (-w, -h, 0, h, w)

LADDER_WS = (0, 1, 4, 8) if not SMOKE else (0, 1)   # +w=4 (budget permits)

# ---- training envelopes --------------------------------------------------------
PRE_STEPS = 3000 if not SMOKE else 60
PRE_CHUNK_S = 175.0 if _USE_GPU else 1700.0        # spec: chunks <= 180 s
PRE_MAX_CHUNKS = 3                                 # spec: <= 3 chunks/root
CE_GATE = 1.65                                     # spec stage-0 gate
CE_PARITY_GATE = 0.05                              # spec stage-0 parity
INSTALL_PZ_LO, INSTALL_PZ_HI = 0.35, 0.75          # spec stage-1 gates
INSTALL_PARITY_GATE = 0.15
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 180.0 if _USE_GPU else 1700.0
EVAL_EVERY = 25 if not SMOKE else 4
NAME_BS, ANCH_BS = 16, 16
WASH_CKPTS = (1, 2, 10, 50) if not SMOKE else (1, 2)
WASH_CAP = 1800.0
COOLDOWN_S = 60.0                                  # dispatch envelope 60-120 s
GPU_WAIT_MAX_S, GPU_POLL_S = 1200.0, 30.0

# ---- registered bar constants (frozen; spec sec 6 verbatim) ---------------------
A_P_ALIVE = 0.15                  # A_P(0) >= +0.15; P-carrier bar
A_A_FACTFREE = 0.10               # A_A(w) < 0.10 at every w
A_A_RIDES = 0.15                  # SLOT-SUCCESSION: A_A >= 0.15
FLAT_RANGE = 0.10                 # NO-SWITCH flat clause
NR_ONSET_MULT = 2.0               # NR >= 2x root
HEADS_EXPR_BAR = 0.30             # heads-by-elimination expression bar
SURG_KILL = 0.60                  # >= 60% drop
SURG_SURVIVE = 0.30               # < 30% drop
SURG_SPARE = 0.10                 # GATE-FREEZE spares: < 10% drop
FLAT_CE_BAR = 0.35                # CE <= +0.35 for any fact-kill claim
WASH_DIE = 0.05                   # g-12 < 0.05 within (1, 2] steps
WASH_SURVIVE = 0.50               # ANY-SURVIVOR: g-12 >= 0.5 at +50
PSI_CONTENT = 0.25                # spine table thresholds (spec sec 4)
PSI_POSITION = 1.0
M_HI = 0.02
M_LO = 0.002
NOVEL0_BAR = 0.10                 # NEAR site-bound bar (registered above)
A_INERT_FRAC = 0.5                # A-arm < 0.5 x P-site (P1)
A_CONTENT_FRAC = 2.0              # A-arm >= 2x P-site (COMPASS-CONTENT)

REGISTERED = {
    "design": "scratch/g4_design.md (committed 2026-09-29 before this file)",
    "P0": "SPINE: the pre-teaching gate class (sec 4 table) predicts the "
          "install carrier AND the w>=1 carrier; DIES on contradiction; "
          "mixed => SPINE-UNSHARP (texture).",
    "P1": "COMPASS-IS-POSITIONAL (committed): NEAR builds P-site content at "
          "rows 5-13, FAR at 137-143; A-slots inert (< 0.5x P-site); row-0 "
          "at/below install baseline; NEAR novel-geometry ~0. Falsifiers: "
          "COMPASS-CONTENT / COMPASS-DEAD.",
    "P2": "CLIFF-ON-P, FLIGHT-TO-HEADS (committed), BOTH roots: A_P(0) >= "
          "+0.15; A_P(1) <= 0; A_A(w) < 0.10 every w; NR onsets at w=1 (>= "
          "2x root); head top-2 share rises w0->w8. Falsifiers: "
          "SLOT-SUCCESSION -> NO-SWITCH-on-dual-while-control-cliffs -> "
          "NO-SWITCH (control).",
    "P3": "DISSOLVES-BY-TWO-STEPS (committed), BOTH roots, ANY carrier: "
          "g-12 < 0.05 within (1,2] steps of the e176N neutral stream. "
          "Falsifier: ANY-SURVIVOR (g-12 >= 0.5 at +50, either root).",
    "P4": "THE HIERARCHY SEGREGATES BY FLOOR (committed): (a) locked fact "
          "dies under D-P-site (>=60% at CE <= +0.35); (b) jitter fact "
          "survives every table surgery (<30% at any CE) and dies under the "
          "N2 knife (>=60% at CE <= +0.35); (c) GATE-FREEZE spares P/head-"
          "carried facts (<10%); (d) D-A-all costs CE (+0.2..+1.0) but "
          "kills no fact.",
    "global": "control cliff? x dual outcome => NECESSITY / LAWS-REAL-"
              "CARRIER-CONTINGENT / INVENTORY-SENSITIVE / SCALE-LINEAGE-"
              "BOUND (spec sec 7 table).",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
}

deviations: list[str] = [
    "w=1's balanced grid is the 3-point {-1,0,1} (e147's recorded deviation; "
    "pool 180 windows vs 300, sampling with replacement, offset 0 in every "
    "grid so the A readout battery stays in-distribution).",
    "The install is the spec's 300-step masked finetune (e109 arm-b recipe "
    "applied at offset 0 from the corpus-only root) — NOT e043's 48-anchor "
    "exposure; the spec stage-1 text is the authority and its gates "
    "[0.35, 0.75] are adjudicated as registered.",
    "The spine is DUAL-ONLY (the control has no gate); its table row is "
    "written to runs/g4/metrics.json BEFORE stage 1 runs (the registration "
    "ordering; T082 lesson).",
    "head top-2 share (W017 dial) = (top-2 positive per-head drops) / (sum "
    "of positive per-head drops) at the g-12 battery; positive head = drop "
    ">= 0.01. The dial is a ratio (share of load), reported with absolute "
    "drops alongside.",
    "carrier(w) heads-by-elimination requires g-12 expression >= 0.30 "
    "(family-2's e157 door shut at 0.198 — a weaker field cannot anchor a "
    "heads call); UNRESOLVED otherwise, numbers reported.",
    "P4's jitter-net primary read is g-12 (the field read); g0 is co-reported "
    "for every cell (the locked net's primary is g0, its site-bound read).",
    "GPU float nondeterminism precedent (e119): fresh GPU trainings carry no "
    "bit-repro expectations; gates are internal (loss finite, steps ran) "
    "plus the registered program gates.",
    "Smoke mode trims: 60-step pretrains, 8-step arms, 2 wash checkpoints, "
    "reduced censuses; nothing adjudicated.",
]

trims: list[str] = []
CKPT_INVENTORY: dict = {}


# ------------------------------------------------------------------ gpu guard

def gate_launch(tag: str) -> None:
    """e119/e143's bounded gate_launch: gpu_ok() double-poll, wait, PARK."""
    t0 = time.time()
    while True:
        if gpu_ok():
            time.sleep(10)
            s2 = gpu_status()
            if gpu_ok():
                log(f"[gpu] launch '{tag}' ok (util {s2['util']:.0f}% temp "
                    f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                    f"{s2['mem_total']:.0f}MB)")
                return
        if time.time() - t0 > GPU_WAIT_MAX_S:
            raise RuntimeError(f"PARK: GPU busy/hot for {GPU_WAIT_MAX_S:.0f}s "
                               f"({gpu_status()}) — refusing '{tag}'")
        time.sleep(GPU_POLL_S)


# ------------------------------------------------------------------ models

class DualGPT(nn.Module):
    """The lab TinyGPT with ONE change: the input floor carries two address
    tables and a gate (spec sec 3). `wpe` attribute name kept so every
    state-dict wpe instrument runs verbatim; blocks/ln_f/lm_head IDENTICAL
    to TinyGPT (pre-LN, causal)."""

    def __init__(self, cfg: Cfg, n_slots: int = N_SLOTS):
        super().__init__()
        self.cfg = cfg
        self.n_slots = n_slots
        self.wte = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.slots = nn.Embedding(n_slots, cfg.n_embd)
        self.gate = nn.Sequential(nn.Linear(2 * cfg.n_embd, GATE_HID),
                                  nn.GELU(), nn.Linear(GATE_HID, n_slots))
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        self.apply(TinyGPT._init)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        wte_x = self.wte(idx)                                   # (B,T,C)
        wpe_t = self.wpe(pos).unsqueeze(0).expand(B, T, self.cfg.n_embd)
        a = torch.softmax(self.gate(torch.cat([wte_x, wpe_t], dim=-1)), -1)
        x = wte_x + self.wpe(pos) + a @ self.slots.weight       # two floors
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n


def build_ctrl() -> TinyGPT:
    return TinyGPT(Cfg(vocab=65, n_layer=N_LAYER, n_head=N_HEAD,
                       n_embd=N_EMBD, block_size=BLOCK))


def build_dual() -> DualGPT:
    return DualGPT(Cfg(vocab=65, n_layer=N_LAYER, n_head=N_HEAD,
                       n_embd=N_EMBD, block_size=BLOCK))


def evl_load(sd: dict, dual: bool):
    m = build_dual() if dual else build_ctrl()
    m.load_state_dict(sd)
    m.eval()
    return m


# ------------------------------------------------------------------ instruments
# PROVENANCE: battery_cell / battery_pz / ce_fixed_cpu / val_windows /
# deleted_wpe / read_fact_at / row_census / finetune_arm are e143's copies
# of the e131 instruments (e068/e109/e113/e116 lineage), model-agnostic.
# The wash is e176n's finetune_freeze arithmetic; the neutral bank is
# e170's. The head knife is common.lesion's c_proj pre-hook (e133/e160).

@torch.no_grad()
def battery_cell(net, ids: torch.Tensor, zid: int, bs=30) -> dict:
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
def battery_pz(net, ids: torch.Tensor, zid: int, bs=30) -> float:
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def ce_fixed_cpu(net, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
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


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2 subtractive row-zero with the e065/e113 confinement gate."""
    out = {k: v.clone() for k, v in sd.items()}
    for r in rows:
        out["wpe.weight"][r] = 0.0
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "confined": bool(changed_rows == sorted(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(rows) * sd["wpe.weight"].shape[1]
                         and changed_rows == sorted(rows) and others)}
    return out, gate


def deleted_slots(sd: dict, ks, mode: str) -> tuple[dict, dict]:
    """The slot-floor surgery (NEW instrument; the deleted_wpe convention
    ported): slots.weight[k] <- 0 ('zero') or <- mean-of-slots ('mean')."""
    out = {k: v.clone() for k, v in sd.items()}
    ks = list(ks)
    if mode == "zero":
        for k in ks:
            out["slots.weight"][k] = 0.0
    elif mode == "mean":
        m = sd["slots.weight"].mean(0)
        for k in ks:
            out["slots.weight"][k] = m
    else:
        raise ValueError(mode)
    d = out["slots.weight"] != sd["slots.weight"]
    changed = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "slots.weight")
    gate = {"slots": ks, "mode": mode, "n_elements_changed": n,
            "expected": len(ks) * sd["slots.weight"].shape[1],
            "changed_slots": changed,
            "confined": bool(changed == sorted(ks)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(ks) * sd["slots.weight"].shape[1]
                         and changed == sorted(ks) and others)}
    return out, gate


GATE_KEYS = ("gate.0.weight", "gate.0.bias", "gate.2.weight", "gate.2.bias")


def gate_freeze_sd(sd_arm: dict, sd_pre_teach: dict) -> tuple[dict, dict]:
    """GATE-FREEZE: swap the gate's weights back to the PRE-TEACHING root
    values at eval — a confined ~18K-param policy edit (spec sec 5)."""
    out = {k: v.clone() for k, v in sd_arm.items()}
    for k in GATE_KEYS:
        out[k] = sd_pre_teach[k].clone()
    n_changed = sum(int((out[k] != sd_arm[k]).sum().item()) for k in GATE_KEYS)
    others = all(torch.equal(out[k], sd_arm[k]) for k in out if k not in GATE_KEYS)
    gate = {"keys": list(GATE_KEYS), "n_elements_changed": n_changed,
            "expected": int(sum(sd_arm[k].numel() for k in GATE_KEYS)),
            "others_bit_identical": bool(others),
            "note": "n_elements_changed == 0 would itself be a finding (the "
                    "gate never moved under teaching); the confinement gate "
                    "is others_bit_identical",
            "pass": bool(others)}
    return out, gate


@torch.no_grad()
def read_fact_at(net, pool_x: torch.Tensor, name_ids, zid: int,
                 addr_row: int, xcol: int, bs=30) -> dict:
    """e131's read_fact_position VERBATIM ARITHMETIC, geometry parameterized."""
    net.eval()
    n_name = len(name_ids)
    per_pos = [[] for _ in range(n_name)]
    onset = []
    for i in range(0, pool_x.shape[0], bs):
        w = pool_x[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(w.shape[0]):
            onset.append(float(pr[k, addr_row, int(zid)]))
            for j in range(n_name):
                per_pos[j].append(
                    float(pr[k, addr_row + j, int(w[k, xcol + j])]))
    onset_t = torch.tensor(onset)
    allp_t = torch.tensor([p for pos in per_pos for p in pos])
    return {"pz_onset_mean": float(onset_t.mean()),
            "pz_onset_median": float(onset_t.median()),
            "pz_onset_frac_ge_0.5": float((onset_t >= 0.5).float().mean()),
            "pname_mean_over7": float(allp_t.mean()),
            "pname_frac_ge_0.5": float((allp_t >= 0.5).float().mean()),
            "per_position_mean": [float(np.mean(pos)) for pos in per_pos]}


def row_census(net, rows, readout, *rargs) -> dict:
    """e139/e140's census VERBATIM (mean-arm / zero-arm / restore)."""
    net.eval()
    base = readout(net, *rargs)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    m_d, z_d = {}, {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d[r] = base - readout(net, *rargs)
        w.copy_(orig); w[r] = 0.0
        z_d[r] = base - readout(net, *rargs)
    w.copy_(orig)
    rows_d = {str(r): {"mean": float(m_d[r]), "zero": float(z_d[r]),
                       "ratio": float(min(m_d[r], z_d[r]) /
                                      max(m_d[r], z_d[r]))
                              if max(m_d[r], z_d[r]) > 0 else 0.0,
                       "strength": float(min(m_d[r], z_d[r])),
                       "content": bool(m_d[r] > 0 and z_d[r] > 0 and
                                       min(m_d[r], z_d[r]) /
                                       max(m_d[r], z_d[r]) >= 0.5)}
              for r in rows}
    assert torch.equal(w, orig), "census failed to restore wpe"
    return {"base_readout": base, "rows": rows_d}


# ----------------------------- NEW readers: the gate / slot instruments

@torch.no_grad()
def gate_table(net: DualGPT) -> torch.Tensor:
    """Gate logits for EVERY (char, position) pair: (V, block, K)."""
    V, B, C = net.cfg.vocab, net.cfg.block_size, net.cfg.n_embd
    wt = net.wte.weight.unsqueeze(1).expand(V, B, C)
    wp = net.wpe.weight.unsqueeze(0).expand(V, B, C)
    return net.gate(torch.cat([wt, wp], -1))


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p.add(1e-12).log() - q.add(1e-12).log())).sum(-1)


@torch.no_grad()
def psi_probe(net: DualGPT, train_ids, n: int, lo: int, hi: int, seed: int):
    """PSI = median KL(a(x,t)||a(x,t+1)) / median KL(a(x,t)||a(x',t)) over
    corpus positions with window-column t in [lo, hi) (spec sec 4)."""
    a = torch.softmax(gate_table(net), -1)          # (V, B, K)
    V = net.cfg.vocab
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(train_ids) - BLOCK - 2, (n,), generator=g)
    cols = torch.randint(lo, hi, (n,), generator=g)
    x = train_ids[starts + cols]
    xp = (x + torch.randint(1, V, (n,), generator=g)) % V   # a random OTHER char
    a_xt = a[x, cols]
    kpos = _kl(a_xt, a[x, cols + 1])
    kcnt = _kl(a_xt, a[xp, cols])
    mp, mc = float(kpos.median()), float(kcnt.median())
    psi = mp / mc if mc > 0 else float("inf")
    return {"psi": psi, "median_kl_position": mp, "median_kl_content": mc,
            "n": n, "col_range": [lo, hi - 1]}


@torch.no_grad()
def slot_usage(net: DualGPT, train_ids, band, n: int, seed: int):
    """M(k): argmax-slot usage mass over corpus tokens at window columns in
    `band` (the read band). Returns (mass vector, detail)."""
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(train_ids) - BLOCK - 1, (n,), generator=g)
    cols = torch.randint(band[0], band[1] + 1, (n,), generator=g)
    toks = train_ids[starts + cols]
    a = torch.softmax(net.gate(torch.cat([net.wte(toks), net.wpe(cols)], -1)),
                      -1)
    am = a.argmax(-1)
    mass = torch.bincount(am, minlength=net.n_slots).float() / n
    order = torch.argsort(mass, descending=True)
    return mass, {"n": n, "band": list(band),
                  "top3_slots": [int(s) for s in order[:3]],
                  "top3_mass": [float(mass[s]) for s in order[:3]],
                  "entropy_nats": float(
                      -(mass.clamp_min(1e-12) * mass.clamp_min(1e-12).log()
                        ).sum())}


@torch.no_grad()
def fact_read_slots(net: DualGPT, ids: torch.Tensor, row: int, k: int = 3):
    """Top-k slots by gate-argmax frequency at the battery's READ row (the
    fact-selected slots; spec sec 5 D-A-top3)."""
    pos = torch.arange(ids.shape[1])
    a = torch.softmax(net.gate(torch.cat([net.wte(ids), net.wpe(pos)
                                          .unsqueeze(0)
                                          .expand(ids.shape[0], -1, -1)], -1)),
                      -1)                           # (N,T,K)
    am = a[:, row, :].argmax(-1)
    cnt = torch.bincount(am, minlength=net.n_slots).float()
    order = torch.argsort(cnt, descending=True)
    return [int(s) for s in order[:k]], {int(s): float(cnt[s] / cnt.sum())
                                         for s in order[:k]}


def slot_census(net: DualGPT, ks, readout, *rargs) -> dict:
    """The e140 census ported to slots: the listed slots jointly <- mean-of-
    slots / <- 0; drop = base - arm; strength = min(mean, zero). Per-slot
    rows co-reported (texture)."""
    net.eval()
    base = readout(net, *rargs)
    w = net.slots.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    ks = list(ks)

    def run(mode):
        w.copy_(orig)
        if mode == "mean":
            for k in ks:
                w[k] = mean_row
        else:
            for k in ks:
                w[k] = 0.0
        d = base - readout(net, *rargs)
        w.copy_(orig)
        return float(d)

    m_d = run("mean")
    z_d = run("zero")
    rows = {}
    for k in ks:
        w.copy_(orig); w[k] = mean_row
        mk = base - readout(net, *rargs)
        w.copy_(orig); w[k] = 0.0
        zk = base - readout(net, *rargs)
        w.copy_(orig)
        rows[str(k)] = {"mean": float(mk), "zero": float(zk),
                        "strength": float(min(mk, zk))}
    assert torch.equal(w, orig), "slot census failed to restore slots"
    return {"base_readout": base, "slots": ks,
            "mean_drop": m_d, "zero_drop": z_d,
            "strength": float(min(m_d, z_d)), "per_slot": rows}


@torch.no_grad()
def head_drops(net, ids: torch.Tensor, zid: int) -> dict:
    """e133 head census via common.lesion's c_proj pre-hook: per-head fact
    drop at the battery read; returns base, (L,H) drops, W017 dials."""
    base = battery_pz(net, ids, zid)
    drops = {}
    for l in range(net.cfg.n_layer):
        for h in range(net.cfg.n_head):
            with lesion(net, "head", l, h):
                pz = battery_pz(net, ids, zid)
            drops[(l, h)] = base - pz
    pos = sorted(v for v in drops.values() if v >= 0.01)
    total = sum(pos)
    top2 = sum(sorted(drops.values(), reverse=True)[:2])
    return {"base_pz": base,
            "drops": {f"L{l}H{h}": float(drops[(l, h)])
                      for l in range(net.cfg.n_layer)
                      for h in range(net.cfg.n_head)},
            "positive_heads": len(pos), "positive_mass": float(total),
            "top2_drop": float(top2),
            "top2_share": float(top2 / total) if total > 1e-9 else 0.0,
            "top2_id": [f"L{l}H{h}" for (l, h) in
                        sorted(drops, key=drops.get, reverse=True)[:2]]}


@torch.no_grad()
def head_ablate(net, ids: torch.Tensor, zid: int, heads) -> float:
    """Zero a SET of heads simultaneously (the e160 knife cells)."""
    ctxs = [lesion(net, "head", l, h) for (l, h) in heads]
    try:
        for c in ctxs:
            c.__enter__()
        return battery_pz(net, ids, zid)
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)


# ------------------------------------------------------------------ training

def pretrain_root(tag: str, model, corpus, ckpt: Path):
    """Chunked ckpt-resume pretrain (train_model defaults: AdamW 1e-3/wd 0.1,
    batch 64, cosine); chunks <= PRE_CHUNK_S, <= PRE_MAX_CHUNKS (spec sec 5)."""
    if SMOKE and ckpt.exists():
        ckpt.unlink()                     # smoke never resumes stale state
    model = model.to(DEV)
    hist, chunks = [], 0
    while chunks < PRE_MAX_CHUNKS:
        gate_launch(f"pretrain_{tag}")
        hist = train_model(model, corpus, steps=PRE_STEPS, lr=1e-3,
                           batch_size=64, max_seconds=PRE_CHUNK_S,
                           eval_every=100 if not SMOKE else 20, ckpt=ckpt)
        chunks += 1
        if hist and hist[-1]["step"] >= PRE_STEPS:
            break
        cooldown(30.0)
    steps = hist[-1]["step"] if hist else 0
    log(f"pretrain[{tag}]: {steps} steps in {chunks} chunk(s), val "
        f"{hist[-1]['val_loss'] if hist else float('nan'):.4f}")
    return {"history_tail": hist[-6:], "steps": steps, "chunks": chunks,
            "val_ce": hist[-1]["val_loss"] if hist else None}


def finetune_arm(tag: str, net0, pool_x: torch.Tensor, pool_mask: torch.Tensor,
                 anchor: torch.Tensor, train_ids: torch.Tensor, r_eval_xy,
                 f_eval_ids, zid: int, seed: int, steps: int | None = None):
    """e143's finetune_arm VERBATIM arithmetic (e109 arm-b / e119-L recipe),
    model-agnostic: batch 32 = 16 install windows + 16 anchors (8 paired +
    8 random); e043 token-level union CE on the name-char targets; AdamW
    (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0; 300 steps / time cap."""
    steps = steps or FT_STEPS
    if DEV.type == "cuda":
        gate_launch(tag)
    net = copy.deepcopy(net0).to(DEV)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, steps + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(DEV)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])],
                        0).to(DEV)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool,
                        device=DEV)
        m[:NAME_BS] = pool_mask[ix].to(DEV)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps or \
                (time.time() - t_start) > FT_TIME_CAP:
            evl.load_state_dict({k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()})
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "p_z_mean": bz["mean_pz"],
                         "frac_argmax_z": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['mean_pz']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed,
            "final_loss": float(loss.item())}


def wash_neutral(tag: str, net0, anchor_neutral: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, gm12_ids, g0_ids,
                 zid: int, seed: int):
    """e176N arm A VERBATIM arithmetic: batch 32 = 16 neutral anchors + 16
    random corpus, full-token CE, AdamW (0.9,0.95) wd 0.1 constant lr 1e-3
    clip 1.0; checkpoints {1,2,10,50}; light CPU evals consume no RNG."""
    ckpt_steps = WASH_CKPTS
    if DEV.type == "cuda":
        gate_launch(tag)
    net = copy.deepcopy(net0).to(DEV)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor_neutral.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    evl = copy.deepcopy(net0)
    for step in range(1, ckpt_steps[-1] + 1):
        aj = torch.randint(n_anc, (16,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (16,), generator=gen)
        anc = anchor_neutral[aj].to(DEV)
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj]).to(DEV)
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in set(ckpt_steps):
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "g_m12_mean_pz": gz["mean_pz"],
                         "g0_mean_pz": gz0["mean_pz"], "ce_r": ce_r,
                         "in_batch_ce": float(loss.item()),
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:3d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f}")
        if (time.time() - t_start) > WASH_CAP:
            trims.append(f"{tag}: wash cap at step {step}")
            break
    net.eval()
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"g4_{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g4", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO))
                            .replace("\\", "/"), **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ stats

def _ranks(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    sv = np.asarray(v, dtype=float)[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


# ------------------------------------------------------------------ main

M: dict = {}          # the incremental metrics tree (flushed every stage)


def flush(rd: Path):
    M["timing"] = {"total_s": round(time.time() - T0, 1)}
    save_json(rd / "metrics.json", E43.jsonable(M))


def main():
    rd = run_dir("g4_smoke" if SMOKE else "g4")
    log(f"G4 THE DUAL-ADDRESS NET (smoke={SMOKE}) -> {rd}")
    log(f"compute: train device {DEV} (gpu_ok at start: {_USE_GPU}), "
        f"cpu threads {torch.get_num_threads()}")
    if not _USE_GPU and not SMOKE:
        deviations.append("GPU parked (gpu_ok() failed at startup or no "
                          "CUDA) — trainings ran CPU-side under the caps; "
                          "reported, adjudication unchanged.")

    M.update({
        "experiment": "g4_dual_address",
        "date": common.now_iso(),
        "design": "scratch/g4_design.md (the authority; bars verbatim)",
        "registered_predictions": REGISTERED,
        "smoke": SMOKE,
        "architecture": {
            "ctrl": {"class": "TinyGPT", "n_layer": N_LAYER, "n_head": N_HEAD,
                     "n_embd": N_EMBD, "block_size": BLOCK,
                     "params": BASE_PARAMS},
            "dual": {"class": "DualGPT", "n_slots": N_SLOTS,
                     "gate": f"Linear({2 * N_EMBD},{GATE_HID})-GELU-"
                             f"Linear({GATE_HID},{N_SLOTS}) on "
                             f"[wte(x_t); wpe(t)] (ONE token, no context)",
                     "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
                     "block_size": BLOCK, "params": DUAL_PARAMS,
                     "delta_vs_ctrl": DUAL_PARAMS - BASE_PARAMS},
            "delta_pct": round(100 * (DUAL_PARAMS - BASE_PARAMS)
                               / BASE_PARAMS, 2),
        },
        "seeds": {"corpus": CORPUS_SEED, "net": NET_SEED,
                  "gate_reseed": GATE_RESEED, "install": INSTALL_SEED,
                  "ladder_arms": ARM_SEED, "near_far": NEAR_FAR_SEED,
                  "wash": WASH_SEED, "neutral_bank": E170_ANCHOR_SEED,
                  "r_eval": R_EVAL_SEED, "psi": PSI_SEED,
                  "slot_usage": SLOT_SEED},
        "deviations": deviations, "trims": trims,
        "ckpt_inventory_ref": "populated at the end (g4_*.pt, gitignored)",
    })

    # ================= protocol rebuild (e043/e143/e147 verbatim) =========
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=CORPUS_SEED)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert train_text.count("ZEPHYRA") == 0 and val_text.count("ZEPHYRA") == 0

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(
        mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    name_ids = corpus.encode(NAME)
    L = len(NAME)
    log(f"protocol rebuilt: install60 {mix}, held30 "
        f"(SPLICE_RNG {E43.SPLICE_RNG})")

    # install pool (e043 windows, offset 0) + anchor bank (e065/e109)
    wins = []
    for p, h in install_occ:
        w = torch.cat([train_ids[p - PRE: p], name_ids,
                       train_ids[p + len(h): p + len(h) + POST_CAP]])
        assert len(w) == BLOCK
        wins.append(w)
    pool0_x = torch.stack(wins)
    pool0_mask = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
    pool0_mask[:, PRE - 1: PRE - 1 + L] = True
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ladder pools (e147 grids)
    pools_x = {0: pool0_x}
    pools_mask = {0: pool0_mask}
    grids = {0: (0,)}
    for w in LADDER_WS:
        if w == 0:
            continue
        grid = offset_grid(w)
        grids[w] = grid
        ws_, ms_ = [], []
        for j in grid:
            for p, h in install_occ:
                pre = train_ids[p - PRE - j: p]
                post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
                win = torch.cat([pre, name_ids, post])
                assert len(win) == BLOCK and torch.equal(
                    win[PRE + j: PRE + j + L], name_ids)
                ws_.append(win)
                m = torch.zeros(BLOCK - 1, dtype=torch.bool)
                m[PRE - 1 + j: PRE - 1 + j + L] = True
                ms_.append(m)
        pools_x[w] = torch.stack(ws_)
        pools_mask[w] = torch.stack(ms_)

    # compass pools (e143 verbatim)
    near_wins = []
    for p, h in install_occ:
        pre = train_ids[p - NEAR_PRE: p]
        post = train_ids[p + len(h): p + len(h) + NEAR_CONT]
        w = torch.cat([pre, name_ids, post])
        assert len(w) == BLOCK
        near_wins.append(w)
    pool_near_x = torch.stack(near_wins)
    pool_near_mask = torch.zeros(len(near_wins), BLOCK - 1, dtype=torch.bool)
    pool_near_mask[:, NEAR_ADDR_ROW: NEAR_ADDR_ROW + L] = True
    pool_far_x, pool_far_mask = pools_x[8] if 8 in pools_x else None, None
    if pool_far_x is None:
        ws_, ms_ = [], []
        for p, h in install_occ:
            pre = train_ids[p - PRE - FAR_J: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - FAR_J]
            ws_.append(torch.cat([pre, name_ids, post]))
            m = torch.zeros(BLOCK - 1, dtype=torch.bool)
            m[FAR_ADDR_ROW: FAR_ADDR_ROW + L] = True
            ms_.append(m)
        pool_far_x, pool_far_mask = torch.stack(ws_), torch.stack(ms_)
    else:
        pool_far_mask = pools_mask[8]

    # batteries (e068 construction)
    bat_ids = {}
    for j in sorted(set(NOVEL_GEO_LADDER) | set(NOVEL_GEO_COMPASS) | {0}):
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    M["protocol"] = {"corpus_seed": CORPUS_SEED, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "grids": {str(w): list(g) for w, g in grids.items()},
                     "batteries": "ctx = train_text[p-PRE-j:p], p(Z) at last "
                                  "position (wpe row 129+j)"}
    M["gates"] = {"G_SPLICE": G_SPLICE}

    # =====================================================================
    # STAGE 0 — ROOTS (control FIRST: the scale gate) + THE SPINE
    # =====================================================================
    log("=" * 78)
    log("STAGE 0 — ROOTS: control first (the scale gate), then dual")
    set_seed(NET_SEED)
    ctrl = build_ctrl()
    assert ctrl.num_params() == BASE_PARAMS, ctrl.num_params()
    set_seed(NET_SEED)
    dual = build_dual()
    assert dual.num_params() == DUAL_PARAMS, dual.num_params()

    pfx = "smoke_" if SMOKE else ""
    ck_ctrl = CKPT_DIR / f"{pfx}g4_ctrl_base.pt"
    ck_dual = CKPT_DIR / f"{pfx}g4_dual_base.pt"
    pre_ctrl = pretrain_root("ctrl", ctrl, corpus, ck_ctrl)
    cooldown(COOLDOWN_S)
    pre_dual = pretrain_root("dual", dual, corpus, ck_dual)

    ce_ctrl = common.estimate_loss(ctrl, corpus, "val", n_batches=20)
    ce_dual = common.estimate_loss(dual, corpus, "val", n_batches=20)
    parity = abs(ce_dual - ce_ctrl)
    G_CE = {"ctrl_val_ce": ce_ctrl, "dual_val_ce": ce_dual,
            "bar": CE_GATE,
            "ctrl_pass": bool(ce_ctrl <= CE_GATE),
            "dual_pass": bool(ce_dual <= CE_GATE)}
    G_PARITY = {"diff": ce_dual - ce_ctrl, "abs_diff": parity,
                "bar": CE_PARITY_GATE, "pass": bool(parity <= CE_PARITY_GATE),
                "remedy_applied": False}
    log(f"stage-0 gates: CE ctrl {ce_ctrl:.4f} / dual {ce_dual:.4f} "
        f"(bar <= {CE_GATE}); parity |diff| {parity:.4f} "
        f"(bar <= {CE_PARITY_GATE}): "
        f"{'PASS' if G_PARITY['pass'] else 'FAIL'}")

    # the ONE gate-only re-seed remedy (spec sec 5): rebuild the dual root
    # (same body seed), re-draw ONLY the gate from GATE_RESEED, re-pretrain
    if not G_PARITY["pass"] and not SMOKE:
        log("parity FAIL -> the ONE gate-only re-seed remedy (seed "
            f"{GATE_RESEED}, recorded); re-pretraining the dual root")
        G_PARITY["remedy_applied"] = True
        if ck_dual.exists():
            ck_dual.unlink()
        set_seed(NET_SEED)
        dual = build_dual()
        torch.manual_seed(GATE_RESEED)
        for m_ in dual.gate.modules():
            if isinstance(m_, nn.Linear):
                nn.init.normal_(m_.weight, mean=0.0, std=0.02)
                if m_.bias is not None:
                    nn.init.zeros_(m_.bias)
        assert dual.num_params() == DUAL_PARAMS
        pre_dual = pretrain_root("dual_reseed", dual, corpus, ck_dual)
        ce_dual = common.estimate_loss(dual, corpus, "val", n_batches=20)
        parity = abs(ce_dual - ce_ctrl)
        G_CE["dual_val_ce"] = ce_dual
        G_CE["dual_pass"] = bool(ce_dual <= CE_GATE)
        G_PARITY.update({"diff": ce_dual - ce_ctrl, "abs_diff": parity,
                         "pass_after_reseed": bool(parity <= CE_PARITY_GATE),
                         "note": "spec remedy: if still violated, record and "
                                 "proceed — the A-floor costs CE and the "
                                 "comparison carries the bound"})
        log(f"post-remedy parity |diff| {parity:.4f}: "
            f"{'PASS' if G_PARITY.get('pass_after_reseed') else 'STILL FAIL '
             '(recorded, proceeding)'}")

    sd_ctrl_base = {k: v.detach().cpu().clone()
                    for k, v in ctrl.state_dict().items()}
    sd_dual_base = {k: v.detach().cpu().clone()
                    for k, v in dual.state_dict().items()}
    save_ckpt("ctrl_base", sd_ctrl_base,
              {"desc": "control root: plain TinyGPT 4L/4H/128d/256, corpus "
                       f"seed {CORPUS_SEED}, net seed {NET_SEED}, "
                       f"{pre_ctrl['steps']} steps", "steps": pre_ctrl["steps"],
               "seed": NET_SEED})
    save_ckpt("dual_base", sd_dual_base,
              {"desc": "dual root: DualGPT (wpe P-floor + 32-slot A-floor + "
                       f"gate), net seed {NET_SEED}"
                       + (" (gate re-seeded 4306)" if G_PARITY["remedy_applied"]
                          else ""),
               "steps": pre_dual["steps"], "seed": NET_SEED})
    M["stage0"] = {
        "ctrl_pretrain": pre_ctrl, "dual_pretrain": pre_dual,
        "gates": {"G_CE": G_CE, "G_PARITY": G_PARITY},
        "params": {"ctrl": int(ctrl.num_params()),
                   "dual": int(dual.num_params())},
    }
    del ctrl, dual

    # ---------------- THE SPINE (dual-only; committed BEFORE stage 1) -----
    log("THE SPINE (pre-teaching; its table is committed to metrics BEFORE "
        "stage 1 runs)")
    netD = evl_load(sd_dual_base, dual=True)
    psi_glob = psi_probe(netD, train_ids, 4000 if not SMOKE else 400,
                         0, BLOCK - 1, PSI_SEED)
    psi_read = psi_probe(netD, train_ids, 2000 if not SMOKE else 200,
                         READ_BAND[0], READ_BAND[1], PSI_SEED + 1)
    mass_g, usage_g = slot_usage(netD, train_ids, (0, BLOCK - 1),
                                 4000 if not SMOKE else 400, SLOT_SEED)
    mass_r, usage_r = slot_usage(netD, train_ids, READ_BAND,
                                 2000 if not SMOKE else 200, SLOT_SEED + 1)
    m_top = usage_r["top3_mass"][0]
    if psi_read["psi"] < PSI_CONTENT and m_top > M_HI:
        spine_row, pred_locked, pred_w = 1, "P", "HEADS"
    elif psi_read["psi"] < PSI_CONTENT and m_top < M_LO:
        spine_row, pred_locked, pred_w = 2, "A", "A"
    elif psi_read["psi"] > PSI_POSITION:
        spine_row, pred_locked, pred_w = 3, "P", "HEADS"
    else:
        spine_row, pred_locked, pred_w = 4, None, None     # SPINE-UNSHARP
    spine = {
        "psi_global": psi_glob, "psi_read_band": psi_read,
        "slot_usage_global": usage_g, "slot_usage_read_band": usage_r,
        "M_top_read_slot": m_top, "M_top3_read_slots": usage_r["top3_mass"],
        "table_row_selected": spine_row,
        "predicted_install_carrier": pred_locked,
        "predicted_w_ge1_carrier": pred_w,
        "committed_before_stage1": True,
        "timestamp": common.now_iso(),
        "note": "spec sec 4 table verbatim thresholds: PSI_read<0.25 & "
                "M>2% => P/HEADS; PSI_read<0.25 & M<0.2% => A/A; PSI_read>1 "
                "=> P/HEADS; else SPINE-UNSHARP (texture, no bars)",
    }
    M["stage0"]["spine"] = spine
    log(f"SPINE: PSI_read {psi_read['psi']:.4f} (global "
        f"{psi_glob['psi']:.4f}) | M(top read slot) {m_top:.4f} "
        f"(top3 {usage_r['top3_slots']}) -> row {spine_row}: "
        f"locked={pred_locked}, w>=1={pred_w}")
    flush(rd)
    del netD

    # =====================================================================
    # STAGE 1 — INSTALL (both roots; e043 splice, spec's 300-step recipe)
    # =====================================================================
    log("=" * 78)
    log("STAGE 1 — INSTALL (both roots)")
    inst = {}
    for rtag, sd0, is_dual in (("ctrl", sd_ctrl_base, False),
                               ("dual", sd_dual_base, True)):
        net0 = evl_load(sd0, dual=is_dual)
        res = finetune_arm(f"install_{rtag}", net0, pool0_x, pool0_mask,
                           anchor, train_ids, r_eval_xy, ids130, zid,
                           INSTALL_SEED)
        sd = res["sd"]
        save_ckpt(f"{rtag}_install", sd,
                  {"desc": f"{rtag} root + 300-step masked install (offset 0, "
                           f"batch 16+16, seed {INSTALL_SEED})",
                   "steps": res["steps_ran"], "seed": INSTALL_SEED,
                   "base": f"runs/checkpoints/g4_{rtag}_base.pt"})
        netI = evl_load(sd, dual=is_dual)
        cen = row_census(netI, CENSUS_ROWS, lambda n: battery_pz(n, ids130, zid))
        A_P = cen["rows"]["129"]["strength"]
        r0 = cen["rows"]["0"]["strength"]
        if is_dual:
            mass_rb, usage_rb = slot_usage(netI, train_ids, READ_BAND,
                                           2000 if not SMOKE else 200,
                                           SLOT_SEED)
            sc = slot_census(netI, usage_rb["top3_slots"],
                             lambda n: battery_pz(n, ids130, zid))
            A_A = sc["strength"]
        else:
            usage_rb, sc, A_A = None, None, None
        pz0 = battery_cell(netI, ids130, zid)["mean_pz"]
        pzh = battery_cell(netI, bat_ids[(0, "held30")], zid)["mean_pz"]
        ce = ce_fixed_cpu(netI, *r_eval_xy)
        hc = head_drops(netI, ids130, zid)
        inst[rtag] = {"traj": res["traj"], "sd": sd, "dual": is_dual,
                      "pz_g0": pz0, "pz_held_g0": pzh, "ce_r": ce,
                      "census": cen, "A_P": A_P, "row0": r0,
                      "slot_usage_read_band": usage_rb,
                      "A_A_census": sc, "A_A": A_A, "head_census": hc,
                      "NR_root": None}
        # NR root baseline (e141) on the installed root
        nr = {}
        for g in NOVEL_GEO_LADDER:
            none_ = battery_pz(netI, bat_ids[(g, "install60")], zid)
            sd_z, gt = deleted_wpe(sd, (0,))
            assert gt["pass"]
            netI.load_state_dict(sd_z)
            dr0 = battery_pz(netI, bat_ids[(g, "install60")], zid)
            netI.load_state_dict(sd)
            nr[g] = {"pz_none": none_, "pz_dr0": dr0, "NR_zero_drop": none_ - dr0}
        inst[rtag]["NR_root"] = nr
        log(f"install[{rtag}]: p(Z)@g0 {pz0:.4f} (gate "
            f"[{INSTALL_PZ_LO},{INSTALL_PZ_HI}]) held {pzh:.4f} CE_R {ce:.4f} "
            f"| A_P {A_P:+.4f} row0 {r0:+.4f} A_A "
            f"{A_A if A_A is None else round(A_A, 4)} | NR_root(g-12) "
            f"{nr[-12]['NR_zero_drop']:+.4f}")
        del net0, netI
        cooldown(COOLDOWN_S)
    G_INST = {
        "ctrl_pz_g0": inst["ctrl"]["pz_g0"], "dual_pz_g0": inst["dual"]["pz_g0"],
        "ctrl_in_band": bool(INSTALL_PZ_LO <= inst["ctrl"]["pz_g0"]
                             <= INSTALL_PZ_HI),
        "dual_in_band": bool(INSTALL_PZ_LO <= inst["dual"]["pz_g0"]
                             <= INSTALL_PZ_HI),
        "parity": abs(inst["dual"]["pz_g0"] - inst["ctrl"]["pz_g0"]),
        "parity_bar": INSTALL_PARITY_GATE,
        "parity_pass": bool(abs(inst["dual"]["pz_g0"] - inst["ctrl"]["pz_g0"])
                            <= INSTALL_PARITY_GATE),
    }
    log(f"G_INST: {G_INST}")

    def classify(A_Pv, A_Av, expr):
        if A_Pv is None:
            return "N/A (control has no A floor)"
        if A_Av is not None and A_Av >= A_A_RIDES and A_Pv < A_P_ALIVE:
            return "A"
        if A_Pv >= A_P_ALIVE and (A_Av is None or A_Av < A_A_RIDES):
            return "P"
        if expr >= HEADS_EXPR_BAR and A_Pv < A_P_ALIVE and \
                (A_Av is None or A_Av < A_A_RIDES):
            return "HEADS"
        return "UNRESOLVED"

    carrier_install = classify(inst["dual"]["A_P"], inst["dual"]["A_A"],
                               inst["dual"]["pz_g0"])
    M["stage1"] = {
        rtag: {k: v for k, v in cell.items() if k != "sd"}
        for rtag, cell in inst.items()}
    M["stage1"]["gates"] = G_INST
    M["stage1"]["carrier_observed_dual"] = carrier_install
    M["stage1"]["carrier_bars"] = (f"P if A_P >= {A_P_ALIVE} & A_A < "
                                   f"{A_A_RIDES}; A if A_A >= {A_A_RIDES} & "
                                   f"A_P < {A_P_ALIVE}; HEADS if expr >= "
                                   f"{HEADS_EXPR_BAR} & both inert")
    log(f"install carrier (dual, observed): {carrier_install} "
        f"[spine predicted {pred_locked}]")
    flush(rd)

    # =====================================================================
    # STAGE 2 — THE SWITCH LADDER (both roots; e147 minimal ladder)
    # =====================================================================
    log("=" * 78)
    log(f"STAGE 2 — THE SWITCH LADDER {{locked, {LADDER_WS[1:]}}}")
    ladder = {}
    for rtag in ("ctrl", "dual"):
        net0 = evl_load(inst[rtag]["sd"], dual=(rtag == "dual"))
        ladder[rtag] = {}
        for w in LADDER_WS:
            res = finetune_arm(f"{rtag}_w{w}", net0, pools_x[w],
                               pools_mask[w], anchor, train_ids, r_eval_xy,
                               ids130, zid, ARM_SEED)
            sd = res["sd"]
            save_ckpt(f"{rtag}_w{w}", sd,
                      {"desc": f"{rtag} installed root + 300-step "
                               f"{'locked' if w == 0 else f'jitter w={w}'} "
                               f"replay (grid {list(grids[w])}, seed "
                               f"{ARM_SEED})",
                       "width": w, "steps": res["steps_ran"],
                       "seed": ARM_SEED,
                       "base": f"runs/checkpoints/g4_{rtag}_install.pt"})
            net = evl_load(sd, dual=(rtag == "dual"))
            cen = row_census(net, CENSUS_ROWS, lambda n: battery_pz(n, ids130,
                                                                    zid))
            A_P = cen["rows"]["129"]["strength"]
            # A_A: top-3 read-band slots <- mean / <- 0 (the spec's census)
            mass_rb, usage_rb = (slot_usage(net, train_ids, READ_BAND,
                                            2000 if not SMOKE else 200,
                                            SLOT_SEED)
                                 if rtag == "dual" else (None, None))
            sc = slot_census(net, usage_rb["top3_slots"],
                             lambda n: battery_pz(n, ids130, zid)) \
                if rtag == "dual" else None
            A_A = sc["strength"] if sc else None
            # CE of the slot-zero surgery (the SLOT-SUCCESSION CE clause)
            A_A_ce = None
            if sc is not None:
                sd_z, gt = deleted_slots(sd, usage_rb["top3_slots"], "zero")
                assert gt["pass"]
                A_A_ce = ce_fixed_cpu(evl_load(sd_z, dual=True), *r_eval_xy)
                del sd_z
            # NR (e141 d_r0) + slot-sink co-report (D-A-top at g-12)
            nr, sink = {}, {}
            for g in NOVEL_GEO_LADDER:
                none_ = battery_pz(net, bat_ids[(g, "install60")], zid)
                sd_z, gt = deleted_wpe(sd, (0,))
                assert gt["pass"]
                net.load_state_dict(sd_z)
                dr0 = battery_pz(net, bat_ids[(g, "install60")], zid)
                net.load_state_dict(sd)
                nr[g] = {"pz_none": none_, "pz_dr0": dr0,
                         "NR_zero_drop": none_ - dr0}
                if rtag == "dual":
                    sd_s, gts = deleted_slots(sd, usage_rb["top3_slots"],
                                              "zero")
                    assert gts["pass"]
                    net.load_state_dict(sd_s)
                    sink[g] = {"pz_da_top": battery_pz(
                        net, bat_ids[(g, "install60")], zid)}
                    net.load_state_dict(sd)
                    del sd_s
            # brake + D-P-site cell at g0
            sd_d, gtd = deleted_wpe(sd, D_ALL)
            assert gtd["pass"]
            net.load_state_dict(sd_d)
            dpsite_g0 = battery_pz(net, ids130, zid)
            net.load_state_dict(sd)
            del sd_d
            ce = ce_fixed_cpu(net, *r_eval_xy)
            pz_g12 = battery_pz(net, bat_ids[(-12, "install60")], zid)
            hc12 = head_drops(net, bat_ids[(-12, "install60")], zid)
            psi_drift = psi_probe(net, train_ids, 1000 if not SMOKE else 100,
                                  READ_BAND[0], READ_BAND[1],
                                  PSI_SEED + 2)["psi"] if rtag == "dual" \
                else None
            ladder[rtag][w] = {
                "traj": res["traj"], "sd": sd, "A_P": A_P, "A_A": A_A,
                "A_A_zero_ce": A_A_ce, "census": cen,
                "row0": cen["rows"]["0"]["strength"],
                "NR": nr, "slot_sink_g12": sink.get(-12),
                "brake_g0": dpsite_g0 - battery_pz(net, ids130, zid),
                "D_P_site_g0": dpsite_g0, "pz_g12": pz_g12,
                "pz_g0": battery_pz(net, ids130, zid),
                "ce_r": ce, "head_census_g12": hc12,
                "psi_read_drift": psi_drift,
                "slot_usage_read_band": usage_rb,
            }
            c = ladder[rtag][w]
            log(f"ladder[{rtag} w={w}]: A_P {A_P:+.4f} A_A "
                f"{A_A if A_A is None else round(A_A, 4)} NR(g-12) "
                f"{nr[-12]['NR_zero_drop']:+.4f} pz_g12 {pz_g12:.3f} "
                f"top2share {hc12['top2_share']:.3f} CE_R {ce:.4f} "
                f"brake {c['brake_g0']:+.4f}")
            del net
            cooldown(COOLDOWN_S)
        del net0

    # ---- P2 adjudication (per root, then global) ----
    p2 = {}
    for rtag in ("ctrl", "dual"):
        cells = ladder[rtag]
        ws = [w for w in LADDER_WS if w in cells]
        A_Ps = {w: cells[w]["A_P"] for w in ws}
        A_As = {w: cells[w]["A_A"] for w in ws if cells[w]["A_A"] is not None}
        NR_root = inst[rtag]["NR_root"][-12]["NR_zero_drop"]
        carriers = {w: classify(cells[w]["A_P"], cells[w]["A_A"],
                                cells[w]["pz_g12"]) for w in ws}
        c1 = bool(A_Ps.get(0, -1) >= A_P_ALIVE)
        c2 = bool(1 in A_Ps and A_Ps[1] <= 0)
        c3 = bool(A_As and all(v < A_A_FACTFREE for v in A_As.values())) \
            if A_As else None
        c4 = bool(1 in cells and
                  cells[1]["NR"][-12]["NR_zero_drop"] >= NR_ONSET_MULT * NR_root)
        c5 = bool(0 in cells and 8 in cells and
                  cells[8]["head_census_g12"]["top2_share"] >
                  cells[0]["head_census_g12"]["top2_share"]) \
            if (0 in cells and 8 in cells) else None
        committed = bool(c1 and c2 and (c3 is not False) and c4
                         and (c5 is not False))
        slot_succ = None
        if A_As:
            for w, v in A_As.items():
                if w >= 1 and v >= A_A_RIDES:
                    ce_z = cells[w]["A_A_zero_ce"]
                    base_ce = cells[w]["ce_r"]
                    if ce_z is not None and ce_z - base_ce <= FLAT_CE_BAR:
                        slot_succ = {"w": w, "A_A": v,
                                     "ce_delta": ce_z - base_ce}
                        break
        flat = bool(max(A_Ps.values()) - min(A_Ps.values()) <= FLAT_RANGE)
        same_carrier = (carriers.get(0) == carriers.get(1)
                        and carriers.get(0) in ("P", "A"))
        no_switch = bool(flat or same_carrier)
        p2[rtag] = {
            "A_P_by_w": A_Ps, "A_A_by_w": A_As, "carriers_by_w": carriers,
            "NR_root_g12": NR_root,
            "NR_by_w": {w: cells[w]["NR"][-12]["NR_zero_drop"] for w in ws},
            "clauses": {"A_P0_ge_015": c1, "A_P1_le_0": c2,
                        "A_A_factfree": c3, "NR_onset_at_w1": c4,
                        "top2_share_rises": c5},
            "committed_branch_fires": committed,
            "slot_succession": slot_succ,
            "no_switch": {"flat": flat, "same_carrier_w0_w1": same_carrier,
                          "fires": no_switch},
        }
        log(f"P2[{rtag}]: committed={committed} clauses={p2[rtag]['clauses']} "
            f"slot_succ={slot_succ} no_switch={no_switch}")

    control_cliff = bool(p2["ctrl"]["committed_branch_fires"])
    if p2["dual"]["committed_branch_fires"]:
        dual_outcome = "COMMITTED (CLIFF-ON-P, FLIGHT-TO-HEADS)"
    elif p2["dual"]["slot_succession"]:
        dual_outcome = "SLOT-SUCCESSION"
    elif control_cliff and p2["dual"]["no_switch"]["fires"]:
        dual_outcome = "NO-SWITCH on dual WHILE control cliffs"
    elif p2["dual"]["no_switch"]["fires"]:
        dual_outcome = "NO-SWITCH"
    else:
        dual_outcome = "TEXTURE"
    if not control_cliff:
        global_verdict = ("SCALE/LINEAGE BOUND (e157 extends): the control "
                          "root fails the cliff at 0.86M/256-ctx — g4's "
                          "verdict is the bound itself; necessity re-opens "
                          "at 2.7M, is NOT re-run")
    elif dual_outcome.startswith("COMMITTED"):
        global_verdict = ("NECESSITY: compass + switch are channel-inventory-"
                          "robust; the carrier is forced by what channels "
                          "can represent")
    elif dual_outcome == "SLOT-SUCCESSION":
        global_verdict = ("LAWS REAL, CARRIER CONTINGENT: the content-address "
                          "exists (new object); switch = flight to any non-"
                          "positional dedicated channel")
    elif dual_outcome.startswith("NO-SWITCH on dual"):
        global_verdict = ("INVENTORY-SENSITIVE: offering a channel "
                          "unpolarizes the competition — contingency of a "
                          "subtler kind")
    else:
        global_verdict = "TEXTURE (numbers reported, no bar fired)"

    M["stage2"] = {
        rtag: {str(w): {k: v for k, v in cell.items() if k != "sd"}
               for w, cell in ladder[rtag].items()}
        for rtag in ladder}
    M["stage2"]["P2"] = p2
    M["stage2"]["control_cliff"] = control_cliff
    M["stage2"]["dual_outcome"] = dual_outcome
    flush(rd)

    # observed w>=1 carrier (w=8 primary; fall back down the ladder)
    w_obs = next((w for w in (8, 4, 1) if w in ladder["dual"]), 1)
    carrier_w = classify(ladder["dual"][w_obs]["A_P"],
                         ladder["dual"][w_obs]["A_A"],
                         ladder["dual"][w_obs]["pz_g12"])

    # P0 adjudication
    if spine_row == 4:
        p0 = {"verdict": "SPINE-UNSHARP",
              "clause": "the pre-teaching gate class fell in the mixed/"
                        "borderline row — texture only, no post-hoc bars "
                        "(spec sec 4)."}
    else:
        match_locked = (carrier_install == pred_locked)
        match_w = (carrier_w == pred_w)
        p0 = {"predicted": {"locked": pred_locked, "w_ge1": pred_w},
              "observed": {"install": carrier_install,
                           f"w{w_obs}": carrier_w},
              "fires": bool(match_locked and match_w),
              "verdict": None}
        p0["verdict"] = "FIRES" if p0["fires"] else "DIES"
        p0["clause"] = ("the architecture predicted its own memory type: "
                        "census carriers match the committed table row."
                        if p0["fires"] else
                        "the census-observed carrier contradicts the table — "
                        "the gate is epiphenomenal to the type decision.")
    M["stage2"]["P0_spine"] = p0
    log(f"P0 SPINE: {p0['verdict']} ({p0.get('clause', '')})")
    flush(rd)

    # =====================================================================
    # STAGE 3 — THE COMPASS (dual primary; control NEAR = P-replication)
    # =====================================================================
    log("=" * 78)
    log("STAGE 3 — THE COMPASS (NEAR rows 5-13 / FAR rows 137-143)")
    compass = {}
    fevals = {"near": pool_near_x[:, :NEAR_PRE],
              "far": pool_far_x[:, :PRE + FAR_J]}
    pools = {"near": (pool_near_x, pool_near_mask),
             "far": (pool_far_x, pool_far_mask)}
    for rtag, arm in (("dual", "near"), ("dual", "far"), ("ctrl", "near")):
        net0 = evl_load(inst[rtag]["sd"], dual=(rtag == "dual"))
        px, pm = pools[arm]
        res = finetune_arm(f"{rtag}_{arm}", net0, px, pm, anchor, train_ids,
                           r_eval_xy, fevals[arm], zid, NEAR_FAR_SEED)
        sd = res["sd"]
        save_ckpt(f"{rtag}_{arm}", sd,
                  {"desc": f"{rtag} installed root + 300-step locked "
                           f"{arm} replay (name x-cols "
                           f"{6 if arm == 'near' else 138}.."
                           f"{12 if arm == 'near' else 144}, seed "
                           f"{NEAR_FAR_SEED})",
                   "steps": res["steps_ran"], "seed": NEAR_FAR_SEED,
                   "base": f"runs/checkpoints/g4_{rtag}_install.pt"})
        net = evl_load(sd, dual=(rtag == "dual"))
        addr = NEAR_ADDR_ROW if arm == "near" else FAR_ADDR_ROW
        xcol = NEAR_Z_XCOL if arm == "near" else FAR_Z_XCOL
        rows = NEAR_ROWS if arm == "near" else FAR_ROWS

        def own_read(n, addr=addr, xcol=xcol, px=px):
            return read_fact_at(n, px, name_ids, zid, addr,
                                xcol)["pz_onset_mean"]

        cen = row_census(net, rows, own_read)
        ctrl_max = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR)
        site_rows = NEAR_SITE_ROWS if arm == "near" else FAR_SITE_ROWS
        site_str = max(cen["rows"][str(r)]["strength"] for r in site_rows)
        site_pos = any(cen["rows"][str(r)]["content"] and
                       cen["rows"][str(r)]["strength"] >=
                       2.0 * max(ctrl_max, 0.0) for r in site_rows)
        a_arm = None
        if rtag == "dual":
            ks, freq = fact_read_slots(net, px[:, :xcol + 1], addr)
            asc = slot_census(net, ks, own_read)
            a_arm = asc["strength"]
        else:
            asc, ks, freq = None, None, None
        own = read_fact_at(net, px, name_ids, zid, addr, xcol)
        novel = {g: battery_pz(net, bat_ids[(g, "install60")], zid)
                 for g in NOVEL_GEO_COMPASS}
        # D-all at g0 (iii)
        sd_d, gtd = deleted_wpe(sd, D_ALL)
        assert gtd["pass"]
        net.load_state_dict(sd_d)
        dall_g0 = battery_pz(net, ids130, zid)
        net.load_state_dict(sd)
        del sd_d
        compass[f"{rtag}_{arm}"] = {
            "traj": res["traj"], "sd": sd, "census": cen,
            "control_max": ctrl_max, "site_rows": list(site_rows),
            "site_strength": site_str, "site_pos": site_pos,
            "row0": cen["rows"]["0"]["strength"],
            "A_arm_strength": a_arm, "A_arm_census": asc,
            "A_arm_slots": ks, "A_arm_slot_freq": freq,
            "own_geometry": own, "novel_geos": novel,
            "D_all_g0": dall_g0,
            "ce_r": ce_fixed_cpu(net, *r_eval_xy),
        }
        c = compass[f"{rtag}_{arm}"]
        log(f"compass[{rtag}/{arm}]: site {site_str:+.4f} (2x-ctrl "
            f"{2 * ctrl_max:.4f}) site_pos {site_pos} row0 {c['row0']:+.4f} "
            f"A-arm {a_arm if a_arm is None else round(a_arm, 4)} own "
            f"{own['pz_onset_mean']:.3f} novel-g12 {novel[-12]:.3f}")
        del net, net0
        cooldown(COOLDOWN_S)

    dn, df = compass["dual_near"], compass["dual_far"]
    r0_base = inst["dual"]["census"]["rows"]["0"]["strength"]
    r0_routed = ladder["dual"][8]["row0"] if 8 in ladder["dual"] else \
        r0_base + 0.2
    midpoint = r0_base + 0.5 * max(r0_routed - r0_base, 0.0)
    novel_mean_near = float(np.mean([dn["novel_geos"][g]
                                     for g in NOVEL_GEO_COMPASS]))
    p1_clauses = {
        "near_site_pos": bool(dn["site_pos"]),
        "far_site_pos": bool(df["site_pos"]),
        "A_inert": bool(dn["A_arm_strength"] is None or
                        dn["A_arm_strength"] < A_INERT_FRAC
                        * max(dn["site_strength"], 1e-9)),
        "row0_at_baseline": bool(dn["row0"] <= midpoint),
        "near_site_bound": bool(novel_mean_near <= NOVEL0_BAR),
    }
    p1_content = bool(dn["A_arm_strength"] is not None and
                      dn["A_arm_strength"] >= A_CONTENT_FRAC
                      * max(dn["site_strength"], 1e-9))
    p1_dead = bool(not dn["site_pos"] and not df["site_pos"]
                   and dn["site_strength"] < 2 * max(dn["control_max"], 0)
                   and df["site_strength"] < 2 * max(df["control_max"], 0))
    if all(p1_clauses.values()):
        p1_verdict = "COMPASS-IS-POSITIONAL (committed branch HELD)"
    elif p1_content:
        p1_verdict = "COMPASS-CONTENT (falsifier: NEAR rides the A floor)"
    elif p1_dead:
        p1_verdict = "COMPASS-DEAD (falsifier: placement inert in the dual net)"
    else:
        p1_verdict = "TEXTURE"
    M["stage3"] = {
        k: {kk: vv for kk, vv in c.items() if kk != "sd"}
        for k, c in compass.items()}
    M["stage3"]["P1"] = {
        "clauses": p1_clauses,
        "row0_install_baseline": r0_base, "row0_routed_ref_w8": r0_routed,
        "midpoint_bar": midpoint, "near_novel_mean": novel_mean_near,
        "compass_content_falsifier": p1_content,
        "compass_dead_falsifier": p1_dead,
        "verdict": p1_verdict,
        "ctrl_near_P_replication": {
            "site_strength": compass["ctrl_near"]["site_strength"],
            "site_pos": compass["ctrl_near"]["site_pos"],
            "note": "the control's NEAR cell replicates the P-side compass "
                    "with NO A floor available"},
    }
    log(f"P1 COMPASS: {p1_verdict} clauses={p1_clauses}")
    flush(rd)

    # =====================================================================
    # STAGE 4 — THE BASIN (e176N arm A on each w=8 consolidated net)
    # =====================================================================
    log("=" * 78)
    log("STAGE 4 — THE BASIN (neutral wash on each w=8 net)")
    # e170's neutral anchor bank VERBATIM
    arng = random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    forbidden = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        if any(f in train_text[s: s + BLOCK + 1] for f in forbidden):
            rejections += 1
            continue
        n_starts.append(s)
    assert len(n_starts) == 16
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {"bank": "e170 neutral construction verbatim (seed "
                        f"{E170_ANCHOR_SEED})", "n": 16,
                "windows_with_host_content": sum(
                    1 for s in n_starts
                    if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
                "pass": True}
    wash = {}
    for rtag in ("dual", "ctrl"):
        w8 = ladder[rtag].get(8) or ladder[rtag][max(ladder[rtag])]
        net0 = evl_load(w8["sd"], dual=(rtag == "dual"))
        cooldown(30.0)
        res = wash_neutral(f"wash_{rtag}", net0, anchor_neutral, train_ids,
                           r_eval_xy, bat_ids[(-12, "install60")],
                           ids130, zid, WASH_SEED)
        step0_gm12 = battery_pz(net0, bat_ids[(-12, "install60")], zid)
        wash[rtag] = {"traj": res["traj"], "sds": res["sds"],
                      "step0_gm12": step0_gm12,
                      "wash_width_used": (8 if 8 in ladder[rtag]
                                          else max(ladder[rtag]))}
        save_ckpt(f"{rtag}_wash{WASH_CKPTS[-1]}", res["sds"][max(res["sds"])],
                  {"desc": f"{rtag} w={wash[rtag]['wash_width_used']} net + "
                           f"{max(res['sds'])}-step neutral wash (e176N arm "
                           f"A, seed {WASH_SEED})",
                   "steps": int(max(res["sds"])), "seed": WASH_SEED})
        log(f"wash[{rtag}]: g-12 {step0_gm12:.4f} -> "
            + " -> ".join(f"+{t['step']} {t['g_m12_mean_pz']:.4f}"
                          for t in res["traj"]))
        del net0
        cooldown(COOLDOWN_S)

    p3 = {}
    for rtag in ("dual", "ctrl"):
        tr = {t["step"]: t["g_m12_mean_pz"] for t in wash[rtag]["traj"]}
        p3[rtag] = {
            "traj": wash[rtag]["traj"], "step0": wash[rtag]["step0_gm12"],
            "dies_by_two_steps": bool(2 in tr and tr[2] < WASH_DIE),
            "gm12_at_2": tr.get(2), "gm12_at_50": tr.get(50),
            "any_survivor": bool(50 in tr and tr[50] >= WASH_SURVIVE),
        }
    p3_verdict = ("DISSOLVES-BY-TWO-STEPS (committed branch HELD)"
                  if all(p3[r]["dies_by_two_steps"] for r in p3)
                  else "ANY-SURVIVOR (falsifier fired)"
                  if any(p3[r]["any_survivor"] for r in p3) else "TEXTURE")
    M["stage4"] = {rtag: {k: v for k, v in wash[rtag].items() if k != "sds"}
                   for rtag in wash}
    M["stage4"]["G_ANCHOR"] = G_ANCHOR
    M["stage4"]["P3"] = {rtag: p3[rtag] for rtag in p2}
    M["stage4"]["P3"]["verdict"] = p3_verdict
    log(f"P3 BASIN: {p3_verdict}")
    flush(rd)

    # =====================================================================
    # STAGE 5 — THE SURGERY (dual root: locked w=0 and jitter w=8 nets)
    # =====================================================================
    log("=" * 78)
    log("STAGE 5 — THE DELETION HIERARCHY (D-P-site / D-P-0 / D-A-top3 / "
        "D-A-all / GATE-FREEZE / N2 knife)")
    surgery = {}
    for label, wkey, prim_ids, prim_name in (
            ("locked", 0, ids130, "g0"),
            ("jitter", 8 if 8 in ladder["dual"] else max(ladder["dual"]),
             bat_ids[(-12, "install60")], "g-12")):
        sd = ladder["dual"][wkey]["sd"]
        net = evl_load(sd, dual=True)
        ce_arm = ladder["dual"][wkey]["ce_r"]
        none_g0 = battery_pz(net, ids130, zid)
        none_gm12 = battery_pz(net, bat_ids[(-12, "install60")], zid)
        none_prim = none_g0 if prim_name == "g0" else none_gm12
        # fact-selected slots at the primary read row
        read_row = 129 if prim_name == "g0" else 117
        ks_fact, ks_freq = fact_read_slots(net, prim_ids, read_row)

        cells = {}

        def run_cell(tag, sd_cell, gate):
            if not gate["pass"]:
                raise RuntimeError(f"surgery gate FAILED {label}/{tag}: {gate}")
            net.load_state_dict(sd_cell)
            pz_g0 = battery_pz(net, ids130, zid)
            pz_m12 = battery_pz(net, bat_ids[(-12, "install60")], zid)
            ce = ce_fixed_cpu(net, *r_eval_xy)
            net.load_state_dict(sd)
            prim = pz_g0 if prim_name == "g0" else pz_m12
            cells[tag] = {"pz_g0": pz_g0, "pz_gm12": pz_m12, "ce_r": ce,
                          "ce_delta": ce - ce_arm,
                          "drop_prim": 1.0 - prim / max(none_prim, 1e-12),
                          "gate": gate}
            log(f"surgery[{label}/{tag}]: pz {prim_name} {prim:.4f} "
                f"(drop {100 * cells[tag]['drop_prim']:.1f}%) dCE "
                f"{ce - ce_arm:+.4f}")

        sd_c, g_c = deleted_wpe(sd, D_ALL)
        run_cell("D-P-site", sd_c, g_c); del sd_c
        sd_c, g_c = deleted_wpe(sd, (0,))
        run_cell("D-P-0", sd_c, g_c); del sd_c          # poison cell, e150
        sd_c, g_c = deleted_slots(sd, ks_fact, "zero")
        run_cell("D-A-top3", sd_c, g_c); del sd_c
        sd_c, g_c = deleted_slots(sd, list(range(N_SLOTS)), "zero")
        run_cell("D-A-all", sd_c, g_c); del sd_c
        sd_c, g_c = gate_freeze_sd(sd, sd_dual_base)
        run_cell("GATE-FREEZE", sd_c, g_c); del sd_c

        # the e160 N2-class knife (escalation; CE measured WITH the set
        # ablated — inside the lesion context)
        hc = head_drops(net, prim_ids, zid)
        ranked = sorted(hc["drops"].items(), key=lambda kv: -kv[1])
        top_heads = [(int(t[1:t.index('H')]), int(t[t.index('H') + 1:]))
                     for t, _ in ranked[:4]]
        knife = {"head_census": hc, "ranking": [t for t, _ in ranked[:8]],
                 "cells": {}}
        for (l, h) in top_heads:
            pz = head_ablate(net, prim_ids, zid, [(l, h)])
            knife["cells"][f"L{l}H{h}"] = {
                "pz_prim": pz,
                "drop": 1.0 - pz / max(none_prim, 1e-12)}
            log(f"surgery[{label}/L{l}H{h}]: drop "
                f"{100 * knife['cells'][f'L{l}H{h}']['drop']:.1f}%")
        for k, n in (("N2", 2), ("N3", 3), ("E4", 4)):
            if n > len(top_heads):
                continue
            set_ = top_heads[:n]
            ctxs = [lesion(net, "head", l, h) for (l, h) in set_]
            for c_ in ctxs:
                c_.__enter__()
            try:
                net.eval()
                pz = battery_pz(net, prim_ids, zid)
                ce_k = ce_fixed_cpu(net, *r_eval_xy)
            finally:
                for c_ in reversed(ctxs):
                    c_.__exit__(None, None, None)
            knife["cells"][k] = {"heads": [f"L{l}H{h}" for l, h in set_],
                                 "pz_prim": pz,
                                 "drop": 1.0 - pz / max(none_prim, 1e-12),
                                 "ce_r": ce_k, "ce_delta": ce_k - ce_arm,
                                 "flat_ce_kill": bool(
                                     1.0 - pz / max(none_prim, 1e-12)
                                     >= SURG_KILL and ce_k - ce_arm
                                     <= FLAT_CE_BAR)}
            log(f"surgery[{label}/{k}]: {set_} drop "
                f"{100 * knife['cells'][k]['drop']:.1f}% dCE "
                f"{ce_k - ce_arm:+.4f} flat-CE kill "
                f"{knife['cells'][k]['flat_ce_kill']}")
        net.load_state_dict(sd)
        # the Z-slot continuation rider (jitter net only): the slot the gate
        # picks on the Z char (gate input = (token, position) only, so it is
        # constant across windows at a given row); does deleting it hit the
        # CONTINUATION while sparing the onset?
        z_rider = None
        if label == "jitter":
            with torch.no_grad():
                a_z = torch.softmax(net.gate(torch.cat(
                    [net.wte.weight[zid].unsqueeze(0),
                     net.wpe.weight[130].unsqueeze(0)], -1)), -1)
                zslot = int(a_z.argmax(-1).item())
                span_base = read_fact_at(net, pool0_x, name_ids, zid, 129, 130)
                sd_c, g_c = deleted_slots(sd, [zslot], "zero")
                net.load_state_dict(sd_c)
                pz_onset_after = battery_pz(net, bat_ids[(-12, "install60")],
                                            zid)
                span_after = read_fact_at(net, pool0_x, name_ids, zid, 129,
                                          130)
                net.load_state_dict(sd)
            z_rider = {"z_slot": zslot, "gate_p_on_zslot": float(a_z[0, zslot]),
                       "baseline_span": span_base, "after_zslot_zero": span_after,
                       "onset_gm12_baseline": none_gm12,
                       "onset_gm12_after": pz_onset_after,
                       "split_carrier": bool(
                           span_after["pname_mean_over7"] <
                           0.5 * span_base["pname_mean_over7"]
                           and pz_onset_after >= 0.8 * none_gm12)}
            del sd_c
            log(f"z-rider: Z-slot {zslot} (p {a_z[0, zslot]:.3f}); span "
                f"{span_base['pname_mean_over7']:.3f} -> "
                f"{span_after['pname_mean_over7']:.3f}, onset g-12 "
                f"{none_gm12:.3f} -> {pz_onset_after:.3f}, split "
                f"{z_rider['split_carrier']}")
        surgery[label] = {
            "w": wkey, "primary_read": prim_name, "ce_arm": ce_arm,
            "none_g0": none_g0, "none_gm12": none_gm12,
            "fact_slots": ks_fact, "fact_slot_freq": ks_freq,
            "cells": cells, "knife": knife, "z_slot_rider": z_rider,
        }
        del net

    lk, jt = surgery["locked"], surgery["jitter"]
    n2_cell = jt["knife"]["cells"].get("N2", {})
    p4 = {
        "a_dpsite_kills_locked": bool(
            lk["cells"]["D-P-site"]["drop_prim"] >= SURG_KILL
            and lk["cells"]["D-P-site"]["ce_delta"] <= FLAT_CE_BAR),
        "b_tables_spare_jitter": bool(
            all(jt["cells"][t]["drop_prim"] < SURG_SURVIVE for t in
                ("D-P-site", "D-A-top3", "D-A-all"))),
        "b_n2_kills_jitter": bool(n2_cell.get("flat_ce_kill")),
        "b_n2_cell": n2_cell,
        "b_escalation_co_report": {k: c for k, c in jt["knife"]["cells"].items()
                                   if k in ("N3", "E4")},
        "c_gate_freeze_spares": bool(
            max(lk["cells"]["GATE-FREEZE"]["drop_prim"],
                jt["cells"]["GATE-FREEZE"]["drop_prim"]) < SURG_SPARE),
        "d_daall_cost": bool(
            0.2 <= jt["cells"]["D-A-all"]["ce_delta"] <= 1.0),
        "d_daall_kills_no_fact": bool(
            jt["cells"]["D-A-all"]["drop_prim"] < SURG_SURVIVE),
    }
    p4_falsified = bool(not p4["b_tables_spare_jitter"]
                        and any(jt["cells"][t]["drop_prim"] >= SURG_KILL
                                and jt["cells"][t]["ce_delta"] <= FLAT_CE_BAR
                                for t in ("D-P-site", "D-A-top3", "D-A-all"))) \
        or not p4["a_dpsite_kills_locked"]
    p4["verdict"] = ("THE HIERARCHY SEGREGATES BY FLOOR (committed HELD)"
                     if (p4["a_dpsite_kills_locked"] and p4["b_tables_spare_jitter"]
                         and p4["b_n2_kills_jitter"] and p4["c_gate_freeze_spares"]
                         and p4["d_daall_cost"] and p4["d_daall_kills_no_fact"])
                     else "FALSIFIED" if p4_falsified else "TEXTURE")
    M["stage5"] = {k: {kk: vv for kk, vv in c.items() if kk != "sd"}
                   for k, c in surgery.items()}
    M["stage5"]["P4"] = p4
    log(f"P4 SURGERY: {p4['verdict']} {p4}")
    flush(rd)

    # =====================================================================
    # GLOBAL VERDICT
    # =====================================================================
    M["verdicts"] = {
        "P0_spine": p0, "P1_compass": M["stage3"]["P1"],
        "P2_switch": {"ctrl": p2["ctrl"], "dual": p2["dual"],
                      "control_cliff": control_cliff,
                      "dual_outcome": dual_outcome},
        "P3_basin": M["stage4"]["P3"], "P4_surgery": p4,
        "global": {"control_cliff": control_cliff,
                   "dual_outcome": dual_outcome,
                   "verdict": global_verdict},
        "secondary_texture": {
            "gate_drift_PSI_read": {
                "spine_pre_teaching": spine["psi_read_band"]["psi"],
                **{f"w{w}": ladder["dual"][w]["psi_read_drift"]
                   for w in ladder["dual"]}},
            "slot_sink_D_A_top_gm12": {
                f"w{w}": ladder["dual"][w]["slot_sink_g12"]
                for w in ladder["dual"]},
            "brake_by_w": {f"w{w}": ladder["dual"][w]["brake_g0"]
                           for w in ladder["dual"]},
            "z_slot_rider": surgery["jitter"]["z_slot_rider"],
        },
        "honesty_reflex": [
            "n=1 seed per cell; the CONTROL root is the within-experiment "
            "replicator for the phase structure; cross-seed replication of "
            "any firing branch is g4R (the e147R precedent), not assumed.",
            "The compass/cliff instruments have NEVER run at 0.86M/256-ctx "
            "before this file (family 1 = 2.7M; family 2's port covered "
            "consolidation + wash only); the control root is the scale gate "
            "and ran FIRST — if it fails the cliff, no dual-net claim reads "
            "as architecture.",
            "Single-lineage scope (T113): the dual/control roots are a NEW "
            "lineage pair (fresh seeds, new architecture); any firing "
            "branch inherits the n=1-family bound until replicated.",
            "The capacity delta is 22,624 params (2.7%) — a capacity "
            "confound is stated as implausible, not proven.",
            "carrier-by-elimination (HEADS) rests on both table censuses "
            "being inert PLUS novel-geometry expression; the N2 knife in "
            "stage 5 is the causal confirmation, and its CE bar is "
            "adjudicated at the registered +0.35.",
            "The pretrain budget is the spec's own (<= 3 x 180 s chunks per "
            "root to >= 3,000 steps); val CE bars are reported against the "
            "registered 1.65 gate without post-hoc extension.",
        ],
    }
    flush(rd)

    # =====================================================================
    # PLOTS
    # =====================================================================
    if not SMOKE:
        plot_all(rd)
    M["ckpt_inventory"] = CKPT_INVENTORY
    flush(rd)
    log(f"outputs: {rd}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_all(rd: Path):
    st2 = M["stage2"]
    ws = sorted(int(w) for w in st2["dual"] if w.isdigit())

    def _g(d, k):                     # int-or-str key tolerance (json/memory)
        return d[k] if k in d else d[str(k)]

    # ---- ladder.png ----
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for k, (dial, title, bar) in enumerate((
            ("A_P", "A_P(w) — the P-floor address key (e140 census)", 0.15),
            ("A_A", "A_A(w) — the A-floor census (top-3 read slots)", 0.10),
            ("NR", "NR(w) — d_r0 at g-12 (e141)", None))):
        ax = axes[0, k]
        for rtag, col in (("ctrl", "gray"), ("dual", "crimson")):
            ys = []
            for w in ws:
                c = st2[rtag][str(w)]
                v = (c["A_P"] if dial == "A_P" else
                     c["A_A"] if c["A_A"] is not None else float("nan"))
                if dial == "NR":
                    v = _g(c["NR"], -12)["NR_zero_drop"]
                ys.append(v)
            ax.plot(ws, ys, "o-", color=col, label=f"{rtag}")
        if bar is not None:
            ax.axhline(bar, ls="--", lw=1, color="navy")
            ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("jitter width w")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    for rtag, col in (("ctrl", "gray"), ("dual", "crimson")):
        ax.plot(ws, [st2[rtag][str(w)]["head_census_g12"]["top2_share"]
                    for w in ws], "o-", color=col, label=rtag)
    ax.set_title("W017 dial: head top-2 share at g-12", fontsize=9.5)
    ax.set_xlabel("w"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for rtag, col in (("ctrl", "gray"), ("dual", "crimson")):
        ax.plot(ws, [st2[rtag][str(w)]["pz_g12"] for w in ws], "o-",
                color=col, label=f"{rtag} g-12")
    ax.set_title("novel-geometry expression (g-12)", fontsize=9.5)
    ax.set_xlabel("w"); ax.legend(fontsize=8)
    ax = axes[1, 2]
    ax.plot(ws, [st2["dual"][str(w)]["psi_read_drift"] for w in ws], "o-",
            color="purple")
    ax.axhline(0.25, ls="--", color="navy", lw=1)
    ax.set_title("gate drift: PSI_read per arm (dual)", fontsize=9.5)
    ax.set_xlabel("w")
    fig.suptitle("G4 — the switch ladder: A_P / A_A / NR, dual vs control",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "ladder.png", dpi=130)
    plt.close(fig)

    # ---- compass.png ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    ax = axes[0]
    for key, col in (("dual_near", "crimson"), ("dual_far", "steelblue"),
                     ("ctrl_near", "gray")):
        cen = M["stage3"][key]["census"]["rows"]
        xs = sorted(int(r) for r in cen)
        ax.plot(xs, [cen[str(r)]["strength"] for r in xs], "o-", ms=3,
                color=col, label=key)
    ax.axvspan(5, 13, color="crimson", alpha=0.07)
    ax.axvspan(137, 143, color="steelblue", alpha=0.10)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row"); ax.set_ylabel("census strength")
    ax.set_title("(i) site content tests (NEAR 5-13 / FAR 137-143)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    vals = [M["stage3"][k]["A_arm_strength"] or 0 for k in
            ("dual_near", "dual_far")]
    ax.bar([0, 1], vals, 0.5, color=["crimson", "steelblue"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["NEAR", "FAR"])
    ax.set_title(f"(ii) A-arm census (NEAR A {vals[0]:.4f} vs P-site "
                 f"{M['stage3']['dual_near']['site_strength']:+.4f})",
                 fontsize=9)
    ax = axes[2]
    for key, col in (("dual_near", "crimson"), ("dual_far", "steelblue")):
        nv = M["stage3"][key]["novel_geos"]
        xs = sorted(nv, key=lambda g: int(g))
        ax.plot([int(g) for g in xs], [nv[g] for g in xs], "o-", color=col,
                label=key)
    ax.set_title("(iv) novel geometry (site-bound NEAR ~0)", fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle(f"G4 — the compass: {M['stage3']['P1']['verdict']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "compass.png", dpi=130)
    plt.close(fig)

    # ---- wash.png ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for rtag, col in (("dual", "crimson"), ("ctrl", "gray")):
        tr = M["stage4"][rtag]["traj"]
        xs = [0] + [t["step"] for t in tr]
        ys = [M["stage4"][rtag]["step0_gm12"]] + \
            [t["g_m12_mean_pz"] for t in tr]
        ax.plot(xs, ys, "o-", color=col, label=f"{rtag} g-12")
        ces = [t["ce_r"] for t in tr]
        ax2 = ax.twinx()
        ax2.plot([t["step"] for t in tr], ces, "--", color=col, alpha=0.4)
        ax2.set_ylabel("CE_R (dashed)", fontsize=8)
    ax.axhline(0.05, ls="--", color="navy", lw=1, label="die bar 0.05")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("neutral-wash steps"); ax.set_ylabel("g-12 p(Z)")
    ax.set_title(f"G4 — the basin: {M['stage4']['P3']['verdict']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rd / "wash.png", dpi=130)
    plt.close(fig)

    # ---- dual_address.png (the deliverable composite) ----
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    ax = axes[0, 0]
    ax.axis("off")
    sp = M["stage0"]["spine"]
    vlines = [
        "G4 — THE DUAL-ADDRESS NET (design scratch/g4_design.md):",
        f"  ctrl {BASE_PARAMS:,} params | dual {DUAL_PARAMS:,} "
        f"(+{DUAL_PARAMS - BASE_PARAMS:,}, "
        f"{M['architecture']['delta_pct']}%)",
        f"  SPINE (pre-teaching): PSI_read {sp['psi_read_band']['psi']:.4f} "
        f"| PSI_global {sp['psi_global']['psi']:.4f}",
        f"  M(top read slot) {sp['M_top_read_slot']:.4f} (top3 "
        f"{sp['slot_usage_read_band']['top3_slots']})",
        f"  table row {sp['table_row_selected']} -> locked="
        f"{sp['predicted_install_carrier']}, w>=1="
        f"{sp['predicted_w_ge1_carrier']}",
        "",
        f"P0 SPINE: {M['stage2']['P0_spine']['verdict']}",
        f"P1 COMPASS: {M['stage3']['P1']['verdict']}",
        f"P2 SWITCH: dual={M['stage2']['dual_outcome']}; control cliff="
        f"{M['stage2']['control_cliff']}",
        f"P3 BASIN: {M['stage4']['P3']['verdict']}",
        f"P4 SURGERY: {M['stage5']['P4']['verdict']}",
        "",
        f"GLOBAL: {M['verdicts']['global']['verdict']}",
    ]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.055, tx, fontsize=8.6, va="top",
                family="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9,
                          edgecolor="gray") if tx.startswith(("GLOBAL", "P"))
                else None)
    axes[0, 1].remove()
    axes[0, 2].remove()
    # reuse ladder panels
    gs = axes[1, 0].get_gridspec()
    for a in axes[1]:
        a.remove()
    sub = fig.add_subplot(2, 3, (4, 6))
    for rtag, col in (("ctrl", "gray"), ("dual", "crimson")):
        sub.plot(ws, [st2[rtag][str(w)]["A_P"] for w in ws], "o-",
                 color=col, label=f"A_P {rtag}")
        sub.plot(ws, [st2[rtag][str(w)]["A_A"] if
                    st2[rtag][str(w)]["A_A"] is not None else float("nan")
                    for w in ws], "s--", color=col, alpha=0.6,
                 label=f"A_A {rtag}")
    sub.axhline(0.15, ls=":", color="navy", lw=1)
    sub.axhline(0, color="k", lw=0.6)
    sub.set_xlabel("jitter width w"); sub.legend(fontsize=7)
    sub.set_title("the ladder: A_P vs A_A (dual vs control)", fontsize=9.5)
    sub2 = fig.add_subplot(2, 3, (2, 3))
    for rtag, col in (("dual", "crimson"), ("ctrl", "gray")):
        tr = M["stage4"][rtag]["traj"]
        sub2.plot([t["step"] for t in tr], [t["g_m12_mean_pz"] for t in tr],
                  "o-", color=col, label=f"{rtag} g-12")
    sub2.axhline(0.05, ls="--", color="navy", lw=1)
    sub2.set_xscale("symlog", linthresh=1)
    sub2.set_xlabel("wash step"); sub2.legend(fontsize=7)
    sub2.set_title("the basin (neutral wash)", fontsize=9.5)
    fig.suptitle("G4 — the dual-address net: compass-and-cliff in a "
                 "two-floor architecture", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "dual_address.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
