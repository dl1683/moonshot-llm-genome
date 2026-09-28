"""E120 — FACT IN CONTEXTS: does a TEACHING signal in SELF-GENERATED contexts
consolidate, and do self-contexts beat other-contexts at matched signal?
(REGISTERED before compute — the tier-1 fallback registered in T074)

Frame: T074 closed the dream road — verbatim own-dream replay (e121) carries
the name (22.5 ZEPHYRA/10k chars) yet does NOT consolidate (post-D-all 0.025,
base floor 0.195); jitter (e109/T064) and deletion pressure (e083/T073) — the
two open roads — both carry an explicit REORGANIZATION signal that verbatim
replay lacks. The discriminating question (dispatch, verbatim): "does a
TEACHING signal in SELF-GENERATED contexts consolidate, and do self-contexts
beat other-contexts at matched signal?"

BASE (frozen): the B43 install line, runs/checkpoints/e082_b43_install.pt
(B43 + e043 exposure verbatim, seed 24331; e082 GATE-0 install-60 p(Z)
0.3198, R1i 0.191; GATE-1 address row r* = 129, NOT retargeted).
REPRODUCTION GATE (G_INST): base install-60 battery p(Z) = 0.3198219 +- 0.005
(e121's exact gate; e091 tolerance convention).

ARMS — matched-exposure fine-tunes from the SAME base net; same step budget
(300 steps), same optimizer (AdamW 0.9/0.95 wd 0.1, constant lr 1e-3, clip
1.0), same batch 32 (16 exposure windows + 16 anchors = 8 paired originals +
8 random corpus, e109 arm-(a) composition verbatim), same draw generator
(seed 12101 for arms a/b/c — slot-matched draws; arm d keeps e109's own
10901). The ONLY difference is the exposure text:
  (a) SELF-CONTEXT-SPLICED: harvest the BASE NET's own free-run dreams with
      e121's harvest machinery VERBATIM (30 held-30 prompts x 4 sampling
      seeds 12110..12113, T=0.8 top_k=40, 126 new tokens; slot k = prompt
      k%30, seed block k//30), then SPLICE the name-fact segments into those
      dream contexts: each window = [held prompt 130][dream 126 with its
      continuation cols 42..73 REPLACED by the fact segment]. FACT SEGMENT
      (slot k) = 12 pre-name chars + ZEPHYRA + 12 post-name chars of install
      window k%60 (the e043 install-window core: the host name replaced by
      ZEPHYRA, local syntax preserved). The fact therefore sits at x-cols
      184..190 (address row 183) in EVERY window — a FIXED novel address,
      deliberately OUTSIDE the 121..137 battery readout band: the battery
      never reads row 183, so post-D-all expression at the readout geometries
      cannot be re-addressing at the splice site, and any in-band growth the
      exposure causes is deleted by e121's D-all rule verbatim. Loss mask =
      the full 126 continuation columns (e121 convention) — the fact carries
      the teaching signal inside the masked span; the dream filler text is
      also under loss (training on own text IS part of "in self contexts").
  (b) CORPUS-CONTEXT-SPLICED: the SAME fact segments spliced at the SAME
      columns into CORPUS continuations — e121's arm-(c) draws verbatim
      (random train-corpus positions, seed 12103, same seed-major slot
      layout). Matched signal (identical fact segments, identical columns,
      identical mask), FOREIGN surroundings. The (a)-(b) contrast is pure
      context-source: own dream text vs corpus text around the same fact.
  (c) VERBATIM-DREAMS (no-signal reference): e121's arm (a) rebuilt
      bit-for-bit — own dreams verbatim (no splice), same mask, same draw
      seed 12101. Rerun in-table (the D-all row set is per-arm), then
      cross-checked against runs/e121/metrics.json: if the install-60
      battery table matches to < 1e-9 the e121 numbers are declared
      BIT-REPRODUCIBLE (reused/validated); else RERUN-PRIMARY (reported).
  (d) JITTERED-REPLAY reference: e109's arm-(a) recipe VERBATIM rebuilt on
      THIS line — the 60 install windows at jittered offsets J = {-8,-4,0,
      +4,+8} (pool of 300 windows; name targets at y-cols 129+j, the only
      masked columns — e109's own registered mask), seed 10901, same 300-step
      budget. e109 itself ran on the e048_repro line (p(Z) 0.5563); this arm
      gives the position-diversity reference at matched budget ON THIS LINE
      (the positive control for the bar-3 reading).

THEN (eval-only, all CPU): delete the address rows — D2-STYLE SUBTRACTIVE
ROW-ZERO on the wpe address rows with the e065/e113 confinement gate (e121's
deleted_wpe VERBATIM; the e042/e083 token-space D2 reset wte[Z]/lm_head[Z] is
NOT used because zeroing lm_head[Z] pins logit-Z at 0 and makes p(Z)
uninformative as a readout). Address sets are PER-NET (e113 convention):
D129 = {129}; D-ALL-ITS-ADDRESSES = {129} + grown rows, grown row r in
{121..137} minus {129} with norm(arm, r) - norm(base, r) > 0.02 (e121's rule
VERBATIM — row 0 never touched; the splice address row 183 is NOT added: the
battery never reads it, and its growth is reported in the wpe probes).
Measured on each net: battery p(Z) at the e113 arm-set geometries (-8,-4,0,
+4,+8; readout at wpe row 129+j), install-60 primary + held-30 report-only,
plus CE_R (60 name-free val windows, e065 seed 26502) for retain health.

REGISTERED BARS (coordinator, verbatim; adjudicated in this priority order):
  1. (a) post-D-all p(Z) >= 0.20 with (b) below by CI separation
     (operationalized: a >= 0.20 at >= 3/5 geometries on install-60 AND a's
     95% bootstrap CI low > b's high at geometry 0)  => SELF-CONTEXT
     ADVANTAGE (the self helps learning when signal is present — W004
     partially rehabilitated as a training prior, not gate).
  2. (a) ~ (b) (overlapping CIs) AND both >= 0.20 (each >= 3/5 geometries)
     => SIGNAL SUFFICES (context irrelevant; the road is the signal).
  3. both < 0.20 (neither reaches 3/5 geometries)  => SIGNAL-IN-CONTEXTS
     INSUFFICIENT (jitter's reorganization — position diversity — is the
     operative ingredient, not teaching per se; adjudicated against the LIVE
     arm-(d) jitter reference on this line: jitter clearing the bar the
     locked-context signal arms miss completes the discrimination).
  4. none of the above => TEXTURE (honest report).
CALIBRATED SECONDARY (always reported): the base net's own post-D-all row is
measured live (this line's unconsolidated D129 floor is 0.1945, e082 A1-zero)
and every arm is additionally read as a CI-referenced delta vs that base;
arm (c)'s reproduction status and arm (d)'s survival are always reported.
COVARIATES (not gates): the harvest's name-rate (e048 counting convention;
e121's own-dream reference is 34 ZEPHYRA / 15,120 chars) and the POST-SPLICE
name inventory per arm — arm (a)'s dream filler retains residual
self-generated ZEPHYRA occurrences (this line dreams the name, T074), so
arm (a) carries slightly more name tokens than arm (b); counted and
reported, NOT stripped (mangling the dream filler would falsify the
self-context treatment). Registered honesty note: this is a signal-matching
imperfection of the "matched signal" clause, bounded by the count.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
torch.set_num_threads(8); e121/e113's exact envelope — the GPU thermal
machinery (180 s caps, double-polls, cooldowns) is void without a GPU; the
CPU reproducibility of arm (c) REQUIRES CPU). Fine-tune wall cap 1500 s each
(e113's CPU envelope). Single-run target ~45 min.

Outputs: runs/e120/{metrics.json, fact_in_contexts.png}.
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e120_fact_in_contexts.py   (E120_SMOKE=1 for smoke)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (coordinator; arm-c repro)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # 8 threads max

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E120_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
BASE_CK = E43.REPO / "runs" / "checkpoints" / "e082_b43_install.pt"
E121_METRICS = E43.REPO / "runs" / "e121" / "metrics.json"

JITTERS = (-8, -4, 0, 4, 8)       # e109's registered jitter set (arm d)
GEO_ORDER = [-8, -4, 0, 4, 8]     # the e113 arm-set geometries
ADDR_BAND = tuple(r for r in range(121, 138))   # e121 grown-row census band
GROWN_D_NORM = 0.02               # e121 registered grown-row inflation threshold
SPLICE_ADDR_ROW = 183             # the splice site's address row (x-col 184)

# splice geometry (registered)
FACT_PRE, FACT_POST = 12, 12      # local install-window context around the name
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42                    # fact-segment onset within the 126 continuation
Z_XCOL = PRE + SPLICE_AT + FACT_PRE            # 184: ZEPHYRA onset x-column

# dream harvest (e121 verbatim)
GEN_T, GEN_TOPK = 0.8, 40         # lab free-run standard
DREAM_SEEDS = (12110, 12111, 12112, 12113)
N_DREAMS_PER_PROMPT = 1 if SMOKE else 4
N_PROMPTS = 8 if SMOKE else 30    # the held-30 prompts (smoke: first 8)
DREAM_LEN = BLOCK - PRE           # 126 new tokens -> 256-token windows
CORP_CONT_SEED = 12103            # e121 arm-(c) corpus-continuation draws (arm-b filler)

# fine-tune envelope (e109 arm (a) / e121 verbatim except CPU caps)
FT_LR = 1e-3
FT_STEPS = 8 if SMOKE else 300
FT_TIME_CAP = 1500.0              # CPU safety cap (e113)
EVAL_EVERY = 2 if SMOKE else 50
NAME_BS = 16                      # exposure windows per step
ANCH_BS = 16                      # anchor windows per step (8 paired + 8 random)
CONS_SEED = 12101                 # e121's draw generator (arms a/b/c slot-matched)
JIT_SEED = 10901                  # e109's registered arm-(a) seed

# batteries / guards
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_INST_REF = 0.3198219250887632   # e082 gate0 final install-60 p(Z) (e121 verbatim)
G_INST_TOL = 0.005
REPRO_TOL = 1e-9                  # arm-(c) bit-reproduction tolerance

# registered bars
BAR_SURVIVE = 0.20
BAR_COLLAPSE = 0.05
BOOT_N = 10000

REGISTERED_BARS = {
    "self_context_advantage": "arm (a) post-D-all p(Z) >= 0.20 (>= 3/5 "
                              "geometries, install-60) with (b) below by CI "
                              "separation (a's bootstrap 95% CI low > b's "
                              "high, geometry 0) => SELF-CONTEXT ADVANTAGE "
                              "(the self helps learning when signal is "
                              "present — W004 partially rehabilitated as a "
                              "training prior, not gate)",
    "signal_suffices": "(a) ~ (b) (overlapping CIs) AND both >= 0.20 (each "
                       ">= 3/5 geometries) => SIGNAL SUFFICES (context "
                       "irrelevant; the road is the signal)",
    "signal_in_contexts_insufficient": "both < 0.20 (neither reaches 3/5 "
                                       "geometries) => SIGNAL-IN-CONTEXTS "
                                       "INSUFFICIENT (jitter's "
                                       "reorganization — position diversity — "
                                       "is the operative ingredient, not "
                                       "teaching per se; adjudicated against "
                                       "the live arm-(d) jitter reference)",
    "texture": "none of the above — honest report",
    "priority": "adjudicated in order 1-3; first match fires",
}

trims: list[str] = []
deviations: list[str] = [
    "Slot numbering: T074 registered this rung as 'the e120-registered tier-1 "
    "fallback (fact spliced into own contexts)'; e121's docstring calls it "
    "'the e120 tier-1 splice fallback'. Outputs under runs/e120/.",
    "Splice operationalization: fact segment = 12 pre + ZEPHYRA + 12 post "
    "chars of install window k%60; onset at continuation col 42; ZEPHYRA at "
    "x-cols 184-190 (address row 183) in EVERY window — a FIXED novel "
    "address, deliberately OUTSIDE the 121..137 readout band. The battery "
    "never reads row 183, so post-D-all expression at the readout geometries "
    "cannot be re-addressing at the splice site; any in-band growth is "
    "deleted by e121's D-all rule verbatim (row 183 growth is REPORTED in "
    "the wpe probes, never deleted). Position diversity = 0 by design — the "
    "discriminator against arm (d).",
    "Loss mask = the full 126 continuation columns (e121 convention) for "
    "arms a/b/c so 'only the exposure text differs' holds exactly across "
    "the matched arms; arm (d) keeps e109's registered name-only mask (its "
    "own recipe, verbatim).",
    "CPU-only (CUDA_VISIBLE_DEVICES=-1, 8 threads): coordinator's preferred "
    "envelope AND a requirement — arm (c)'s bit-reproduction of e121's "
    "arm (a) (CPU, 8 threads) only holds on CPU. The GPU 180 s cap / "
    "double-poll / cooldown machinery is void without a GPU; per-fine-tune "
    "CPU wall cap 1500 s (e113 precedent) so the registered 300 steps "
    "complete.",
    "Arm (c) is RERUN in-table (the D-all row set is per-arm) and then "
    "cross-checked against runs/e121/metrics.json on every shared "
    "install-60 battery cell; verdict BIT-REPRODUCIBLE (< 1e-9 max diff) "
    "validates reusing e121's numbers, else RERUN-PRIMARY.",
    "Arm (d): e109's arm-(a) recipe verbatim (jittered install windows, "
    "name-only masks, seed 10901) rebuilt on the B43 line — e109 itself ran "
    "on the e048_repro line (install p(Z) 0.5563 vs this line's 0.3198); "
    "the live on-line reference replaces citing e109's shape.",
    "Covariate honesty: arm (a)'s dream filler retains residual "
    "self-generated ZEPHYRA occurrences (this line dreams the name, T074), "
    "so arm (a) carries slightly more name tokens than arm (b); counted "
    "and reported, NOT stripped (mangling the filler would falsify the "
    "self-context treatment).",
    "e121's arm (c) (corpus continuations verbatim, no fact: post-D-all g0 "
    "0.0001) is cited from e121 as the no-signal/no-self reference — not "
    "rerun (identical machinery; not a registered e120 arm).",
]


# ------------------------------------------------------------------ instruments

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30,
                 keep_per_ctx=False) -> dict:
    """e068-style battery on CPU: p(Z) at the last position over contexts."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    out = {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
           "std_pz": float(p.std()),
           "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
           "frac_argmax_z": amax / ids.shape[0]}
    if keep_per_ctx:
        out["pz_per_ctx"] = [float(v) for v in p.tolist()]
    return out


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    """e065 ce_fixed (CPU)."""
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


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2-style subtractive row-zero on wpe rows; confinement gate (e065
    G_SURG convention: exact element count, row confinement, everything else
    bit-identical). Row 0 is NEVER touched by construction of the row sets."""
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


@torch.no_grad()
def free_run_batch(net: TinyGPT, prompt_ids: torch.Tensor, n_new: int,
                   seed: int, temperature: float = GEN_T,
                   top_k: int = GEN_TOPK) -> torch.Tensor:
    """Batched free-run (common.generate semantics: T, top-k, per-row
    independent multinomial) — e121 verbatim. Returns (n_prompts, PRE+n_new)."""
    net.eval()
    torch.manual_seed(seed)
    idx = prompt_ids.clone()
    for _ in range(n_new):
        logits, _ = net(idx[:, -net.cfg.block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx


def dream_covariates(windows: torch.Tensor, itos, pre=PRE) -> dict:
    """e048 counting convention over the continuation region of each window."""
    texts = ["".join(itos[int(i)] for i in w[pre:].tolist()) for w in windows]
    chars = sum(len(t) for t in texts)
    zephyra = sum(t.count("ZEPHYRA") for t in texts)
    zephs = sum(t.count("ZEPH") for t in texts)
    z_chars = sum(t.count("Z") for t in texts)
    return {
        "n_windows": len(windows), "n_chars": chars,
        "zephyra_count": zephyra, "zeph_count": zephs,
        "z_char_count": z_chars,
        "windows_with_zephyra": sum(1 for t in texts if "ZEPHYRA" in t),
        "per_10k_chars": {
            "ZEPHYRA": 1e4 * zephyra / max(chars, 1),
            "ZEPH": 1e4 * zephs / max(chars, 1),
            "Z_chars": 1e4 * z_chars / max(chars, 1)},
        "sample_continuation_heads": texts[:3],
    }


# ------------------------------------------------------------------ fine-tune

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """e109 arm-(a) / e121 fine-tune recipe VERBATIM (composition/seed/loss),
    CPU: batch 32 = 16 exposure windows from the pool + 16 anchors (8 paired
    + 8 random); token-level union CE over the exposure mask + full anchor CE;
    AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0; 300 steps."""
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(CPU)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(CPU)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=CPU)
        m[:NAME_BS] = pool_mask[ix].to(CPU)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
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
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


# ------------------------------------------------------------------ splice

def build_fact_segments(install_occ, train_text, encode) -> torch.Tensor:
    """The name-fact segments: 12 pre + ZEPHYRA + 12 post chars of each
    install window (host name replaced by ZEPHYRA — the e043 install core)."""
    segs = []
    for p, h in install_occ:
        s = (train_text[p - FACT_PRE: p] + NAME
             + train_text[p + len(h): p + len(h) + FACT_POST])
        if len(s) != FACT_LEN:
            raise RuntimeError(f"fact segment len {len(s)} != {FACT_LEN}")
        segs.append(encode(s))
    return torch.stack(segs)                     # (60, 31)


def splice_pool(prompt_ids: torch.Tensor, filler: torch.Tensor,
                fact_segs: torch.Tensor) -> torch.Tensor:
    """Window = [prompt 130][filler[0:42] + fact[k%60] + filler[73:126]].
    Slot layout seed-major (slot k = prompt k%30) — e121's pools verbatim."""
    n = filler.shape[0]
    if filler.shape[1] != DREAM_LEN:
        raise RuntimeError(f"filler len {filler.shape[1]} != {DREAM_LEN}")
    fs = fact_segs[torch.arange(n) % fact_segs.shape[0]]
    cont = torch.cat([filler[:, :SPLICE_AT], fs,
                      filler[:, SPLICE_AT + FACT_LEN:]], 1)
    if cont.shape[1] != DREAM_LEN:
        raise RuntimeError(f"spliced continuation len {cont.shape[1]}")
    pr = torch.stack([prompt_ids[k % prompt_ids.shape[0]] for k in range(n)])
    return torch.cat([pr, cont], 1)              # (n, 256)


def splice_name_inventory(pool_x: torch.Tensor, itos) -> dict:
    """Post-splice name inventory: ZEPHYRA occurrences in the fact span
    (x-cols 184..190) vs residual occurrences elsewhere in the window."""
    zid_txt = NAME
    in_span = 0
    residual = 0
    total = 0
    for w in pool_x:
        t = "".join(itos[int(i)] for i in w.tolist())
        total += t.count(zid_txt)
        in_span += int(t[Z_XCOL: Z_XCOL + len(NAME)] == zid_txt)
        residual += t.count(zid_txt) - int(t[Z_XCOL: Z_XCOL + len(NAME)] == zid_txt)
    return {"n_windows": len(pool_x), "total_zephyra": total,
            "in_fact_span": in_span, "residual_in_filler": residual,
            "residual_per_10k_chars": 1e4 * residual / max(len(pool_x) * BLOCK, 1)}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e120_smoke" if SMOKE else "e120")
    log(f"E120 FACT IN CONTEXTS (self-spliced vs corpus-spliced vs verbatim vs "
        f"jitter; smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda visible = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e082/e109/e121 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

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
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 "
        f"(SPLICE_RNG {E43.SPLICE_RNG})")

    # batteries per geometry (e113/e121 construction verbatim): ctx =
    # train_text[p-PRE-j:p], readout p(Z) at the last position (row 129+j).
    bat_ids = {}
    for j in GEO_ORDER:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval_ids = bat_ids[(0, "install60")]          # the G_INST battery

    # CE_R eval bank (e065 verbatim, seed 26502)
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # held prompts (the dream-harvest contexts; never trained in any form)
    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])
    log(f"held prompts: {prompt_ids.shape[0]} x {PRE} chars")

    # anchor bank (e065/e109/e121 verbatim): first 16 install-position
    # ORIGINAL host windows (incumbent continuations, no ZEPHYRA)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- base net + instrument gate
    net0 = load_cpu(BASE_CK)
    evl = copy.deepcopy(net0)
    bz0 = battery_cell(evl, f_eval_ids, zid)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF,
              "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST base installed battery p(Z) {bz0['mean_pz']:.6f} "
        f"(ref {G_INST_REF:.4f}): {'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("base net/battery mismatch vs e082 (gate failed)")
    ce_r0 = ce_fixed_cpu(evl, *r_eval_xy)
    log(f"CE_R (60 name-free val windows): {ce_r0:.4f}")

    # ---------------- dream harvest (e121 machinery verbatim) + covariates
    wins = []
    for s in DREAM_SEEDS[:N_DREAMS_PER_PROMPT]:
        out = free_run_batch(net0, prompt_ids, DREAM_LEN, seed=s)
        wins.append(out)
        log(f"  harvest[own] seed {s}: {out.shape[0]} dreams x "
            f"{DREAM_LEN} new tokens")
    dream_ids = torch.cat(wins)                    # (n_slots, 256), seed-major
    cov_dream = dream_covariates(dream_ids, itos)
    c = cov_dream
    log(f"  harvest[own]: ZEPHYRA {c['zephyra_count']} in {c['n_chars']} "
        f"chars ({c['per_10k_chars']['ZEPHYRA']:.1f}/10k) | Z chars "
        f"{c['z_char_count']} | windows w/ name "
        f"{c['windows_with_zephyra']}/{c['n_windows']}")
    if not SMOKE:
        log("  harvest crosscheck vs e121 (own): "
            f"{'MATCH' if c['zephyra_count'] == 34 else 'DRIFT'} "
            f"(e121 ref 34 ZEPHYRA / 15120 chars)")

    # ---------------- corpus continuations (e121 arm-(c) draws verbatim)
    g = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - DREAM_LEN - 1,
                        (N_DREAMS_PER_PROMPT, len(prompts)), generator=g)
    cont = torch.stack([train_ids[s: s + DREAM_LEN] for s in src.flatten()])
    if cont.shape[0] != dream_ids.shape[0]:
        raise RuntimeError("slot count mismatch dream vs corpus filler")

    # ---------------- fact segments + the two spliced pools
    fact_segs = build_fact_segments(install_occ, train_text, corpus.encode)
    log(f"fact segments: {tuple(fact_segs.shape)} "
        f"({FACT_PRE}+{len(NAME)}+{FACT_POST}); splice onset cont-col "
        f"{SPLICE_AT}; ZEPHYRA x-cols {Z_XCOL}..{Z_XCOL + len(NAME) - 1} "
        f"(address row {SPLICE_ADDR_ROW})")
    dream_fill = dream_ids[:, PRE:]                # (n_slots, 126) self text
    pool_a_x = splice_pool(prompt_ids, dream_fill, fact_segs)
    pool_b_x = splice_pool(prompt_ids, cont, fact_segs)
    inv_a = splice_name_inventory(pool_a_x, itos)
    inv_b = splice_name_inventory(pool_b_x, itos)
    for nm, inv in (("a_self", inv_a), ("b_corpus", inv_b)):
        log(f"  pool[{nm}]: {inv['n_windows']} windows | ZEPHYRA in-span "
            f"{inv['in_fact_span']} | residual-in-filler "
            f"{inv['residual_in_filler']}")
    # splice geometry gate: the fact sits exactly where registered, in both
    name_ids_t = corpus.encode(NAME)
    G_GEO = {"z_xcols": [Z_XCOL, Z_XCOL + len(NAME) - 1],
             "a_ok": bool(torch.equal(pool_a_x[0, Z_XCOL:Z_XCOL + len(NAME)],
                                      name_ids_t)),
             "b_ok": bool(torch.equal(pool_b_x[0, Z_XCOL:Z_XCOL + len(NAME)],
                                      name_ids_t)),
             "a_all": bool(all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)],
                                           name_ids_t) for w in pool_a_x)),
             "b_all": bool(all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)],
                                           name_ids_t) for w in pool_b_x))}
    G_GEO["pass"] = bool(G_GEO["a_all"] and G_GEO["b_all"])
    if not G_GEO["pass"]:
        raise RuntimeError(f"splice geometry gate FAILED: {G_GEO}")
    log(f"G_GEO splice geometry: ZEPHYRA at x-col {Z_XCOL} in all windows of "
        f"a/b: {'PASS' if G_GEO['pass'] else 'FAIL'}")

    # ---------------- arm (d) pool: e109 jittered install windows verbatim
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        wins_j = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name_ids_t, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"window len {len(w)} != {BLOCK} at offset {j}")
            wins_j.append(w)
        jit_x[j] = torch.stack(wins_j)
        m = torch.zeros(len(wins_j), BLOCK - 1, dtype=torch.bool)
        m[:, PRE - 1 + j: PRE - 1 + j + len(NAME)] = True
        jit_mask[j] = m
    pool_d_x = torch.cat([jit_x[j] for j in JITTERS])          # (300, 256)
    pool_d_mask = torch.cat([jit_mask[j] for j in JITTERS])

    # ---------------- arm (c) pool: own dreams VERBATIM (e121 arm-a repro)
    pool_c_x = dream_ids

    # pools + masks (full-continuation mask, e121 convention)
    m = torch.zeros(pool_a_x.shape[0], BLOCK - 1, dtype=torch.bool)
    m[:, PRE - 1:] = True                          # y-cols 129..254 (126 cols)
    pool_mask = m
    log(f"pools: a {tuple(pool_a_x.shape)} | b {tuple(pool_b_x.shape)} | "
        f"c {tuple(pool_c_x.shape)} | d {tuple(pool_d_x.shape)} | a/b/c mask "
        f"cols {int(pool_mask[0].sum())}/window, d mask cols "
        f"{int(pool_d_mask[0].sum())}/window")

    # ---------------- the four matched-exposure fine-tunes
    arms = {}
    for tag, pool, mask, seed, desc in (
            ("a_self_ctx_spliced", pool_a_x, pool_mask, CONS_SEED,
             f"own dreams with the name-fact spliced at x-col {Z_XCOL} "
             f"(self surroundings, teaching signal)"),
            ("b_corpus_ctx_spliced", pool_b_x, pool_mask, CONS_SEED,
             "corpus continuations with the SAME fact segments at the SAME "
             "columns (foreign surroundings, matched signal)"),
            ("c_verbatim_dreams", pool_c_x, pool_mask, CONS_SEED,
             "own dreams VERBATIM, no splice (e121 arm-a bit-repro; the "
             "no-signal reference)"),
            ("d_jitter_replay", pool_d_x, pool_d_mask, JIT_SEED,
             f"e109 arm-(a) recipe on this line: install windows jittered "
             f"{list(JITTERS)}, name-only mask (position-diversity "
             f"reference)")):
        log(f"ARM {tag}: {FT_STEPS} steps, lr {FT_LR}, batch 32, seed {seed} "
            f"— {desc}")
        res = finetune_arm(tag, net0, pool, mask, anchor, train_ids,
                           r_eval_xy, f_eval_ids, zid, seed)
        res["desc"] = desc
        arms[tag] = res

    # ---------------- nets to measure: base + the four arms
    nets = {"base_no_ft": {"sd": net0.state_dict(),
                           "desc": "e082_b43_install reference (no fine-tune)"}}
    for tag, res in arms.items():
        nets[tag] = {"sd": res["sd"], "desc": res["desc"]}

    # ---------------- wpe probes + grown-row detection (e121 rule verbatim)
    w_base = net0.state_dict()["wpe.weight"]
    probe_rows = (0,) + ADDR_BAND + (SPLICE_ADDR_ROW,)
    wpe_probes = {}
    addr_sets = {}
    for tag, n in nets.items():
        w = n["sd"]["wpe.weight"]
        wpe_probes[tag] = {str(r): {"norm": float(w[r].norm()),
                                    "base_norm": float(w_base[r].norm()),
                                    "d_norm": float(w[r].norm() - w_base[r].norm())}
                           for r in probe_rows}
        grown = tuple(r for r in ADDR_BAND
                      if r != 129 and w[r].norm() - w_base[r].norm() > GROWN_D_NORM)
        addr_sets[tag] = {"d129": (129,), "d_all_addresses": (129,) + grown,
                          "grown_rows": list(grown)}
        log(f"wpe[{tag}]: grown rows (d_norm > {GROWN_D_NORM}) = "
            f"{list(grown) or 'none'} | d129 d_norm "
            f"{wpe_probes[tag]['129']['d_norm']:+.3f} | splice-row "
            f"{SPLICE_ADDR_ROW} d_norm "
            f"{wpe_probes[tag][str(SPLICE_ADDR_ROW)]['d_norm']:+.3f}")

    # ---------------- deletion x geometry battery (e121 machinery verbatim)
    table = {}                     # (net, deletion, geometry, battery) -> cell
    gates_surg = {}
    evl = copy.deepcopy(net0)
    for tag, n in nets.items():
        for dl_name, rows in (("none", ()), ("d129", addr_sets[tag]["d129"]),
                              ("d_all_addresses", addr_sets[tag]["d_all_addresses"])):
            if dl_name == "none":
                sd_del, gate = {k: v.clone() for k, v in n["sd"].items()}, \
                    {"rows": [], "pass": True, "note": "no deletion"}
            else:
                sd_del, gate = deleted_wpe(n["sd"], rows)
            gates_surg[(tag, dl_name)] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl_name}: {gate}")
            evl.load_state_dict(sd_del)
            for j in GEO_ORDER:
                for bt in ("install60", "held30"):
                    table[(tag, dl_name, j, bt)] = battery_cell(
                        evl, bat_ids[(j, bt)], zid,
                        keep_per_ctx=(bt == "install60"))
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            table[(tag, dl_name, "ce_r", "-")] = {"ce_r": ce_r}
            log(f"net {tag:22s} deletion {dl_name:17s} (rows "
                f"{list(rows) or '-'}): install60 "
                + " ".join(f"g{g_:+d} {table[(tag, dl_name, g_, 'install60')]['mean_pz']:.3f}"
                           for g_ in GEO_ORDER)
                + f" | CE_R {ce_r:.4f}")

    # ---------------- bootstrap CIs (post-D-all, geometry 0, install-60)
    def boot_ci(per_ctx):
        arr = np.asarray(per_ctx, dtype=np.float64)
        rg = np.random.default_rng(12105)
        idx = rg.integers(0, len(arr), size=(BOOT_N if not SMOKE else 200,
                                             len(arr)))
        means = arr[idx].mean(1)
        return {"mean": float(arr.mean()),
                "lo": float(np.percentile(means, 2.5)),
                "hi": float(np.percentile(means, 97.5))}

    ci = {tag: boot_ci(table[(tag, "d_all_addresses", 0, "install60")]
                       ["pz_per_ctx"]) for tag in nets}
    ci_d129 = {tag: boot_ci(table[(tag, "d129", 0, "install60")]
                            ["pz_per_ctx"]) for tag in nets}
    for tag in nets:
        log(f"post-D-all g0 install60 [{tag}]: mean {ci[tag]['mean']:.4f} "
            f"95% CI [{ci[tag]['lo']:.4f}, {ci[tag]['hi']:.4f}]")

    # report-only crosscheck: base post-D129 g0 vs e082's A1-zero reference
    X_D129_REF = 0.1945319922024038      # e082 A1 own-row destroy (zero)
    x_d129 = {"this_run": ci_d129["base_no_ft"]["mean"], "e082_ref": X_D129_REF,
              "diff": ci_d129["base_no_ft"]["mean"] - X_D129_REF,
              "note": "same net, same battery, same surgery — device-exact "
                      "crosscheck of the deletion/battery machinery "
                      "(report-only)"}
    log(f"crosscheck base post-D129 g0: {x_d129['this_run']:.4f} "
        f"(e082 A1 ref {X_D129_REF:.4f}, diff {x_d129['diff']:+.4f})")

    # ---------------- arm-(c) bit-reproduction crosscheck vs e121
    repro = {"mode": "smoke (skipped)" if SMOKE else "full",
             "e121_file": str(E121_METRICS)}
    if not SMOKE and E121_METRICS.exists():
        e121 = json.load(open(E121_METRICS, encoding="utf-8"))
        e121_bt = e121["battery_table"]
        pairs = [("c_verbatim_dreams", "a_own_dreams"),
                 ("base_no_ft", "base_no_ft")]
        diffs = []
        per_arm = {}
        for mine, theirs in pairs:
            d = 0.0
            n_cells = 0
            for dl in ("none", "d129", "d_all_addresses"):
                for g_ in GEO_ORDER:
                    k = f"{theirs}__{dl}__g{g_:+d}__install60"
                    if k in e121_bt:
                        d = max(d, abs(e121_bt[k]["mean_pz"]
                                       - table[(mine, dl, g_, "install60")]["mean_pz"]))
                        n_cells += 1
            per_arm[mine] = {"e121_arm": theirs, "n_cells": n_cells,
                             "max_abs_diff_mean_pz": d}
            diffs.append(d)
        max_diff = max(diffs)
        repro.update({
            "per_arm": per_arm,
            "max_abs_diff": max_diff,
            "tol": REPRO_TOL,
            "harvest_zephyra_this_run": cov_dream["zephyra_count"],
            "harvest_zephyra_e121": 34,
            "harvest_match": bool(cov_dream["zephyra_count"] == 34),
            "verdict": "BIT-REPRODUCIBLE (e121 numbers reused/validated)"
                       if max_diff < REPRO_TOL else "RERUN-PRIMARY (deviation "
                       "reported; this run's table is primary)",
        })
        log(f"arm-(c) repro vs e121: max |diff| {max_diff:.3e} over "
            f"{sum(v['n_cells'] for v in per_arm.values())} cells -> "
            f"{repro['verdict']}")
    elif not SMOKE:
        repro["verdict"] = "e121 metrics.json MISSING — RERUN-PRIMARY"

    # ---------------- adjudication (registered, in priority order)
    def counts(tag, dl, bt="install60"):
        vals = [table[(tag, dl, g_, bt)]["mean_pz"] for g_ in GEO_ORDER]
        return {"per_geometry": {f"g{g_:+d}": table[(tag, dl, g_, bt)]["mean_pz"]
                                 for g_ in GEO_ORDER},
                "max": max(vals), "min": min(vals),
                "n_ge_020": sum(v >= BAR_SURVIVE for v in vals),
                "n_le_005": sum(v <= BAR_COLLAPSE for v in vals)}

    cd = {(tag, dl): counts(tag, dl)
          for tag in nets for dl in ("none", "d129", "d_all_addresses")}
    held = {(tag, dl): counts(tag, dl, "held30")
            for tag in nets for dl in ("none", "d129", "d_all_addresses")}

    a_surv = cd[("a_self_ctx_spliced", "d_all_addresses")]["n_ge_020"] >= 3
    b_surv = cd[("b_corpus_ctx_spliced", "d_all_addresses")]["n_ge_020"] >= 3
    d_surv = cd[("d_jitter_replay", "d_all_addresses")]["n_ge_020"] >= 3
    a_gt_b_ci = bool(ci["a_self_ctx_spliced"]["lo"] > ci["b_corpus_ctx_spliced"]["hi"])
    a_b_overlap = not (ci["a_self_ctx_spliced"]["lo"] > ci["b_corpus_ctx_spliced"]["hi"] or
                       ci["b_corpus_ctx_spliced"]["lo"] > ci["a_self_ctx_spliced"]["hi"])

    if a_surv and a_gt_b_ci:
        fired = "SELF-CONTEXT ADVANTAGE"
    elif a_b_overlap and a_surv and b_surv:
        fired = "SIGNAL SUFFICES"
    elif (not a_surv) and (not b_surv):
        fired = "SIGNAL-IN-CONTEXTS INSUFFICIENT"
    else:
        fired = "TEXTURE"

    jitter_reading = {
        "d_survives": bool(d_surv),
        "d_max_g": cd[("d_jitter_replay", "d_all_addresses")]["max"],
        "d_n_ge_020": cd[("d_jitter_replay", "d_all_addresses")]["n_ge_020"],
        "note": ("arm (d) is the live position-diversity reference at matched "
                 "budget on this line; if bar 3 fired AND (d) clears the "
                 "0.20 bar, the operative ingredient is position diversity "
                 "(jitter's reorganization), not the teaching signal per se; "
                 "if (d) also fails, budget/line weakness is the honest "
                 "confound — report as texture"),
    }

    # calibrated secondary: deltas vs the live base post-D-all reference
    cal = {}
    for tag in ("a_self_ctx_spliced", "b_corpus_ctx_spliced",
                "c_verbatim_dreams", "d_jitter_replay"):
        cal[tag] = {"post_dall_g0": ci[tag]["mean"],
                    "base_post_dall_g0": ci["base_no_ft"]["mean"],
                    "delta_vs_base": ci[tag]["mean"] - ci["base_no_ft"]["mean"],
                    "ci_excludes_base_mean": bool(
                        ci[tag]["lo"] > ci["base_no_ft"]["mean"] or
                        ci[tag]["hi"] < ci["base_no_ft"]["mean"])}

    adjudication = {
        "counts_post_d_all": {t: cd[(t, "d_all_addresses")] for t in nets},
        "counts_post_d129": {t: cd[(t, "d129")] for t in nets},
        "counts_pre_none": {t: cd[(t, "none")] for t in nets},
        "held30_post_d_all": {t: held[(t, "d_all_addresses")] for t in nets},
        "a_survives": bool(a_surv), "b_survives": bool(b_surv),
        "d_survives": bool(d_surv),
        "a_gt_b_ci_separated": a_gt_b_ci, "a_b_ci_overlap": bool(a_b_overlap),
        "bootstrap_ci_g0_post_dall": ci, "bootstrap_ci_g0_post_d129": ci_d129,
        "calibrated_vs_base": cal,
        "arm_c_reproduction": repro,
        "jitter_reference": jitter_reading,
        "name_rate_covariate": {
            "dream_harvest_pre_splice": cov_dream,
            "post_splice": {"a_self_ctx_spliced": inv_a,
                            "b_corpus_ctx_spliced": inv_b},
            "note": "arm (a)'s dream filler retains residual self-generated "
                    "ZEPHYRA (T074: this line dreams the name); arm (b)'s "
                    "corpus filler has zero (corpus contains no ZEPHYRA) — "
                    "the in-span signal is identical (120), the residual is "
                    "the covariate, reported not stripped"},
        "fired": fired,
        "headline": (f"post-D-all install-60 g0: self-spliced "
                     f"{ci['a_self_ctx_spliced']['mean']:.3f} "
                     f"[{ci['a_self_ctx_spliced']['lo']:.3f},"
                     f"{ci['a_self_ctx_spliced']['hi']:.3f}] | corpus-spliced "
                     f"{ci['b_corpus_ctx_spliced']['mean']:.3f} "
                     f"[{ci['b_corpus_ctx_spliced']['lo']:.3f},"
                     f"{ci['b_corpus_ctx_spliced']['hi']:.3f}] | verbatim "
                     f"{ci['c_verbatim_dreams']['mean']:.3f} | jitter(ref) "
                     f"{ci['d_jitter_replay']['mean']:.3f} | base "
                     f"{ci['base_no_ft']['mean']:.3f} -> {fired}"),
    }
    log("=" * 78)
    log(f"E120 VERDICT: {fired}")
    log(f"  (a) self-ctx-spliced post-D-all: max "
        f"{cd[('a_self_ctx_spliced', 'd_all_addresses')]['max']:.3f} n>=0.20 "
        f"{cd[('a_self_ctx_spliced', 'd_all_addresses')]['n_ge_020']}/5 g0 "
        f"{ci['a_self_ctx_spliced']['mean']:.3f} "
        f"[{ci['a_self_ctx_spliced']['lo']:.3f},{ci['a_self_ctx_spliced']['hi']:.3f}]")
    log(f"  (b) corpus-ctx-spliced post-D-all: max "
        f"{cd[('b_corpus_ctx_spliced', 'd_all_addresses')]['max']:.3f} n>=0.20 "
        f"{cd[('b_corpus_ctx_spliced', 'd_all_addresses')]['n_ge_020']}/5 g0 "
        f"{ci['b_corpus_ctx_spliced']['mean']:.3f} "
        f"[{ci['b_corpus_ctx_spliced']['lo']:.3f},{ci['b_corpus_ctx_spliced']['hi']:.3f}]")
    log(f"  (c) verbatim dreams post-D-all: g0 "
        f"{ci['c_verbatim_dreams']['mean']:.3f} (repro: {repro.get('verdict', '-')})")
    log(f"  (d) jitter replay post-D-all: max "
        f"{cd[('d_jitter_replay', 'd_all_addresses')]['max']:.3f} n>=0.20 "
        f"{cd[('d_jitter_replay', 'd_all_addresses')]['n_ge_020']}/5 | "
        f"base post-D-all g0 {ci['base_no_ft']['mean']:.3f}")
    log(f"  a>b CI-separated: {a_gt_b_ci} | a survives: {a_surv} | "
        f"b survives: {b_surv} | d survives: {d_surv}")
    log(f"  covariate: harvest {cov_dream['zephyra_count']} ZEPHYRA / "
        f"{cov_dream['n_chars']} chars | post-splice in-span a/b "
        f"{inv_a['in_fact_span']}/{inv_b['in_fact_span']} | residual filler "
        f"a {inv_a['residual_in_filler']} b {inv_b['residual_in_filler']}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e120_fact_in_contexts",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("T074's registered tier-1 fallback; coordinator "
                         "dispatch. Docstring written before compute; bars "
                         "verbatim below"),
        "registered_bars": REGISTERED_BARS,
        "question": ("verbatim dreams fail (no signal) and jitter succeeds "
                     "(reorganization signal) — does a TEACHING signal in "
                     "SELF-GENERATED contexts consolidate, and do "
                     "self-contexts beat other-contexts at matched signal?"),
        "base": f"runs/checkpoints/{BASE_CK.name} (e082 B43 install line)",
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries": list(GEO_ORDER),
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j); install-60 "
                                             "primary, held-30 report-only",
                     "held_prompts": f"{N_PROMPTS} held-30 contexts x "
                                     f"{N_DREAMS_PER_PROMPT} dreams",
                     "dream_harvest": {"T": GEN_T, "top_k": GEN_TOPK,
                                       "seeds": list(DREAM_SEEDS[:N_DREAMS_PER_PROMPT]),
                                       "new_tokens": DREAM_LEN},
                     "splice": {"fact_pre": FACT_PRE, "fact_post": FACT_POST,
                                "fact_len": FACT_LEN, "onset_cont_col": SPLICE_AT,
                                "z_xcols": [Z_XCOL, Z_XCOL + len(NAME) - 1],
                                "z_y_cols": [Z_XCOL - 1, Z_XCOL - 1 + len(NAME) - 1],
                                "address_row": SPLICE_ADDR_ROW,
                                "fact_source": "install window k%60 "
                                               "(12 pre + ZEPHYRA + 12 post)",
                                "note": "fixed novel address, outside the "
                                        "121..137 readout band; battery never "
                                        "reads row 183; in-band growth "
                                        "deleted by e121's D-all verbatim"}},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST, "G_GEO": G_GEO,
                  "G_SURG": {f"{k[0]}/{k[1]}": v for k, v in gates_surg.items()},
                  "ce_r_base": ce_r0, "xcheck_base_post_d129": x_d129},
        "fine_tune": {"lr": FT_LR, "steps": FT_STEPS, "time_cap_s": FT_TIME_CAP,
                      "batch": f"{NAME_BS} exposure + {ANCH_BS} anchor "
                               f"({ANCH_BS // 2} paired + {ANCH_BS // 2} random)",
                      "loss": "e043 token-level union CE (a/b/c: exposure "
                              "mask = the 126 continuation cols; d: e109's "
                              "name-only mask)",
                      "optimizer": "AdamW (0.9,0.95) wd 0.1 clip 1.0 constant lr",
                      "seeds": {"abc": CONS_SEED, "d": JIT_SEED},
                      "device": "cpu",
                      "arms": {t: {"desc": arms[t]["desc"],
                                   "steps_ran": arms[t]["steps_ran"],
                                   "traj": arms[t]["traj"]}
                               for t in arms}},
        "harvest_covariates": cov_dream,
        "splice_name_inventory": {"a_self_ctx_spliced": inv_a,
                                  "b_corpus_ctx_spliced": inv_b},
        "e121_citations": {
            "verbatim_own_dreams_post_dall_g0": 0.0248,
            "corpus_continuations_verbatim_post_dall_g0": 0.0001,
            "base_post_dall_g0": 0.1945,
            "note": "e121 values cited for context; arm (c) reproduced "
                    "in-run (see adjudication.arm_c_reproduction)"},
        "wpe_row_probes": wpe_probes,
        "address_sets": {t: {"d129": list(v["d129"]),
                             "d_all_addresses": list(v["d_all_addresses"]),
                             "grown_rows": v["grown_rows"]}
                         for t, v in addr_sets.items()},
        "battery_table": {f"{t}__{dl}__g{g_:+d}__{bt}":
                          table[(t, dl, g_, bt)]
                          for t in nets for dl in ("none", "d129",
                                                   "d_all_addresses")
                          for g_ in GEO_ORDER for bt in ("install60", "held30")},
        "ce_r_table": {f"{t}__{dl}": table[(t, dl, "ce_r", "-")]["ce_r"]
                       for t in nets for dl in ("none", "d129",
                                                "d_all_addresses")},
        "adjudication": adjudication,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: fact_in_contexts.png
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    net_order = ["base_no_ft", "a_self_ctx_spliced", "b_corpus_ctx_spliced",
                 "c_verbatim_dreams", "d_jitter_replay"]
    net_lbl = {"base_no_ft": "base (no fine-tune)",
               "a_self_ctx_spliced": "(a) SELF-ctx spliced",
               "b_corpus_ctx_spliced": "(b) CORPUS-ctx spliced",
               "c_verbatim_dreams": "(c) verbatim dreams",
               "d_jitter_replay": "(d) jitter replay (ref)"}
    net_col = {"base_no_ft": "tab:gray", "a_self_ctx_spliced": "crimson",
               "b_corpus_ctx_spliced": "darkorange",
               "c_verbatim_dreams": "steelblue",
               "d_jitter_replay": "seagreen"}
    xs = np.arange(len(GEO_ORDER))

    # (0,0) MAIN: arm x geometry, pre (none) vs post (D-all) deletion
    ax = axes[0, 0]
    bw = 0.8 / 10
    k = 0
    for t in net_order:
        pre = [table[(t, "none", g_, "install60")]["mean_pz"] for g_ in GEO_ORDER]
        post = [table[(t, "d_all_addresses", g_, "install60")]["mean_pz"] for g_ in GEO_ORDER]
        ax.bar(xs + (k - 4.5) * bw, pre, bw * 0.92, color=net_col[t],
               alpha=0.35, edgecolor="k", linewidth=0.3,
               label=f"{net_lbl[t]}: pre-del (light)")
        ax.bar(xs + (k - 3.5) * bw, post, bw * 0.92, color=net_col[t],
               edgecolor="k", linewidth=0.4,
               label=f"post-D-all (solid)")
        k += 2
    ax.axhline(BAR_SURVIVE, color="seagreen", ls="--", lw=1.1,
               label=f"survive bar {BAR_SURVIVE}")
    ax.axhline(BAR_COLLAPSE, color="gray", ls=":", lw=1.1,
               label=f"collapse bar {BAR_COLLAPSE}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"g{g_:+d}\n(row {129 + g_})" for g_ in GEO_ORDER],
                       fontsize=8)
    ax.set_ylabel("battery p(Z) (install-60)")
    ax.set_ylim(0, 1.05)
    ax.set_title("MAIN: arm x geometry, pre-deletion vs post-D-all "
                 "(light = pre, solid = post)", fontsize=10)
    ax.legend(fontsize=6.0, ncol=2, loc="upper left")

    # (0,1) post-D-all g0 with bootstrap CIs + D129 comparison
    ax = axes[0, 1]
    xs2 = np.arange(len(net_order))
    for k, t in enumerate(net_order):
        v = ci[t]["mean"]
        e = [[v - ci[t]["lo"]], [ci[t]["hi"] - v]]
        ax.bar(k - 0.19, v, 0.36, color=net_col[t], edgecolor="k", lw=0.4,
               yerr=e, capsize=3, error_kw={"lw": 1.0})
        v2 = ci_d129[t]["mean"]
        ax.bar(k + 0.19, v2, 0.36, color=net_col[t], alpha=0.45,
               edgecolor="k", lw=0.4)
        ax.text(k - 0.19, ci[t]["hi"] + 0.012, f"{v:.3f}", ha="center",
                fontsize=7)
        ax.text(k + 0.19, ci_d129[t]["mean"] + 0.012, f"{v2:.3f}", ha="center",
                fontsize=7)
    ax.axhline(BAR_SURVIVE, color="seagreen", ls="--", lw=1.1)
    ax.axhline(BAR_COLLAPSE, color="gray", ls=":", lw=1.1)
    ax.set_xticks(xs2)
    ax.set_xticklabels([net_lbl[t].replace(" (", "\n(") for t in net_order],
                       fontsize=7.5)
    ax.set_ylabel("post-deletion p(Z), geometry 0 (install-60)")
    ax.set_ylim(0, 1.0)
    ax.set_title("solid = post-D-ALL (95% bootstrap CI) | light = post-D129",
                 fontsize=10)

    # (1,0) covariates + wpe d_norm
    ax = axes[1, 0]
    axr = ax.twinx()
    tags_h = ["harvest (pre-splice)", "pool a (self)", "pool b (corpus)"]
    rates = [cov_dream["per_10k_chars"]["ZEPHYRA"],
             1e4 * inv_a["total_zephyra"] / max(inv_a["n_windows"] * BLOCK, 1),
             1e4 * inv_b["total_zephyra"] / max(inv_b["n_windows"] * BLOCK, 1)]
    xs3 = np.arange(3)
    ax.bar(xs3, rates, 0.5, color=["purple", "crimson", "darkorange"],
           edgecolor="k", lw=0.4)
    for k, r in enumerate(rates):
        ax.text(k, r + max(rates) * 0.02, f"{r:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xs3)
    ax.set_xticklabels(tags_h, fontsize=8)
    ax.set_ylabel("ZEPHYRA per 10k chars")
    ax.set_title("name-rate covariate (a carries residual dream names; "
                 "b's corpus filler has none)", fontsize=9)
    rows_x = np.arange(len(ADDR_BAND))
    for k, t in enumerate(net_order):
        dn = [wpe_probes[t][str(r)]["d_norm"] for r in ADDR_BAND]
        axr.plot(rows_x, dn, "o-", ms=3, lw=1.0, color=net_col[t],
                 label=f"{net_lbl[t]} d_norm")
    axr.axhline(GROWN_D_NORM, color="k", ls=":", lw=0.9)
    axr.set_ylabel("wpe row d_norm vs base (band 121-137)")
    axr.legend(fontsize=6.0, loc="upper right")

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    a_ = cd[("a_self_ctx_spliced", "d_all_addresses")]
    b_ = cd[("b_corpus_ctx_spliced", "d_all_addresses")]
    d_ = cd[("d_jitter_replay", "d_all_addresses")]
    lines = [
        "E120 FACT IN CONTEXTS — base e082_b43_install (p(Z) 0.3198 gate "
        f"{'PASS' if G_INST['pass'] else 'FAIL'})",
        f"fact spliced at x-col {Z_XCOL} (address row {SPLICE_ADDR_ROW}, outside "
        "read band; fixed = no position diversity)",
        "",
        f"harvest: {cov_dream['zephyra_count']} ZEPHYRA / {cov_dream['n_chars']} "
        f"chars ({cov_dream['per_10k_chars']['ZEPHYRA']:.1f}/10k) | in-span "
        f"a/b {inv_a['in_fact_span']}/{inv_b['in_fact_span']} | residual "
        f"a {inv_a['residual_in_filler']} b {inv_b['residual_in_filler']}",
        "",
        "post-D-all install-60 (95% CI, geometry 0):",
        f"  (a) self    {ci['a_self_ctx_spliced']['mean']:.3f} "
        f"[{ci['a_self_ctx_spliced']['lo']:.3f}, {ci['a_self_ctx_spliced']['hi']:.3f}] "
        f"max_g {a_['max']:.3f} n>=.20 {a_['n_ge_020']}/5",
        f"  (b) corpus  {ci['b_corpus_ctx_spliced']['mean']:.3f} "
        f"[{ci['b_corpus_ctx_spliced']['lo']:.3f}, {ci['b_corpus_ctx_spliced']['hi']:.3f}] "
        f"max_g {b_['max']:.3f} n>=.20 {b_['n_ge_020']}/5",
        f"  (c) verbatim {ci['c_verbatim_dreams']['mean']:.3f} "
        f"(e121 repro: {repro.get('verdict', 'smoke')})",
        f"  (d) jitter   {ci['d_jitter_replay']['mean']:.3f} max_g {d_['max']:.3f} "
        f"n>=.20 {d_['n_ge_020']}/5 (position-diversity ref)",
        f"  base         {ci['base_no_ft']['mean']:.3f} "
        f"[{ci['base_no_ft']['lo']:.3f}, {ci['base_no_ft']['hi']:.3f}] "
        "(live no-ft reference)",
        "",
        f"a>b CI-separated: {a_gt_b_ci} | a survives: {a_surv} | "
        f"b survives: {b_surv} | d survives: {d_surv}",
        "address sets (D-all): " + " | ".join(
            f"{t.split('_', 1)[0]}: "
            f"{len(addr_sets[t]['d_all_addresses'])} rows"
            + (f" {{{','.join(map(str, addr_sets[t]['d_all_addresses'][:5]))}...}}"
               if len(addr_sets[t]['d_all_addresses']) > 5 else
               f" {{{','.join(map(str, addr_sets[t]['d_all_addresses']))}}}"
               if addr_sets[t]['d_all_addresses'] else " {}")
            for t in net_order),
        "",
        f"FIRED BAR: {fired}",
    ]
    ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=7.6,
            family="monospace", transform=ax.transAxes,
            bbox=dict(facecolor="lightyellow", alpha=0.94, edgecolor="gray"))

    fig.suptitle(f"E120 — FACT IN CONTEXTS: teaching signal in self vs corpus "
                 f"contexts (matched) -> {fired}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "fact_in_contexts.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'fact_in_contexts.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
