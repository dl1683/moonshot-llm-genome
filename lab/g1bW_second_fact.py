"""G1BW — THE WALL'S MUSEUM TEST: can a walled net install a SECOND fact?
(R56's critic-forced experiment; scratch/r56_critic.md section 1 attack 1c +
the KILLER CONTROL, lines 54-64 — the spec source, VERBATIM bars below.)

CONTEXT. g1b/g1bR's WALL: commit(R=0.7) + hard L2 projection holds fact A's
battery-channel ruler (g-12 ~0.9) through +300 wash while same-draw controls
die by +50 (n=3 wash seeds, one lineage). The critic's two verified attacks:
(i) channels of the SAME fact collapse inside the ball (the wpe band readout
~0.001 from +2; g1bR W1 old-band row0 strength 0.762 -> 0.615 over +2..+50);
(ii) SEQUENTIAL MEMORY was never tested — without it "memory is made
architectural" risks reducing to "the organism was frozen with its probe
intact". This cell is the critic's own KILLER CONTROL.

THE CELL (critic spec VERBATIM, lines 54-64): W1 machinery on the seed-10907
lineage; commit + 50 wash steps, then install a SECOND fact B (fresh host
windows + fresh name token, e043-Dmix install VERBATIM, 300 steps) UNDER the
projection (the ball's anchor stays at A's commit). Reads: A-ruler (the g1b
fact battery), B-ruler (same battery form for B), CE_r (root-stream CE), and
free-run completions of BOTH facts.
  REFERENCE LEG (co-reported, not gating): the same B-install on the UNWALLED
  washed control at +50 (same wash seed) — the naive-washed net's B-install is
  the capacity reference for interpreting MUSEUM vs ZERO-SUM.
  ZERO-COMPUTE RIDER: the free-run expression battery on the EXISTING
  runs/checkpoints/g1bR_W1_10907_s300.pt (verified present; NOT retrained).

===================== REGISTERED BARS (frozen; VERBATIM from the critic) =====
  SPLINT-REFUTED: "fires if B installs (B-ruler >= 0.7) with A >= 0.5 — a real
      memory architecture; the wall is not a splint."
  MUSEUM: "fires if B fails (<= 0.27) at healthy CE — the wall is a splint;
      rescope the claim to 'the wall freezes the organism; memory survives
      freezing'."
  ZERO-SUM: "fires if B installs and A dies (A < 0.5) — displacement inside the
      ball kills; the ball is a single-exhibit museum by geometry."
  Channel rider (co-reported): A's wpe-band channel and row0 strength at every
      checkpoint (the critic's collapse channels) — so the fold can scope which
      channels of A the wall protected during B's install.

OPERATIONALIZATIONS (registered here, BEFORE the run; the critic's prose made
measurable with the arc's own instruments; no bar shopping):
  - A-ruler  = the g1b fact battery, g-12 form (mean p(Z) at the last position
    of the install-60 contexts at offset -12) — the WALL-HOLDS claim's own
    ruler. A >= 0.5 / A < 0.5 read at the END of B's install (final state);
    the full per-checkpoint table is co-reported.
  - B-ruler  = the SAME battery form for fact B (mean p(Q) at the last
    position of B's install-60 contexts at offset -12). B installs = B-ruler
    >= 0.7 (the critic's own threshold); read at the END.
  - healthy CE = CE_r (the seed-26502 root-stream val bank, g1's instrument)
    <= root CE_r + 0.30 at the END — g1's CE_NOISE_BAR, the arc's established
    walled-arm health bound.
  - if B lands strictly in (0.27, 0.7), or B fails at UNhealthy CE, NO bar
    fires — reported as AMBIGUOUS (texture with numbers), never re-tuned.

GATES (construction fidelity; any failure => verdict TEXTURE, bars reported
as measured but nothing adjudicated):
  G-ROOT    root loads bit-exact from e131_consolidated_e113.pt, g-12 >= 0.78.
  G-BITROOT the walled root's body is bit-identical to theta0 (anchors copies).
  G-CTRL    the reference leg's wash kills A by +50 (g1bR's own +50 reading
            0.0028; same seed 10907).
  G-WASHREP the walled wash's +50 light g-12 reproduces g1bR's recorded 0.9157
            within 0.05 (device-float fuzz; g1bR migrated CPU mid-run).
  G-PIN     the walled leg's raw displacement vs theta0 (the anchor) <= R +
            1.5 at EVERY checkpoint of BOTH phases (wash + install).
  G-BFRESH  B's name has 0 train/val occurrences; B's windows are disjoint
            from A's install/held windows; B's battery base rate at the root
            (mean p(Q) over B's g-12 contexts) <= 0.05 (the slot is
            incumbent-owned before the install).
  G-INPUTS  per-step install batches bit-identical (md5) between the walled
            and reference legs (the wall is the only delta).
  G-ANCHOR  at every install checkpoint the anchor tensors are still bit-equal
            to theta0 and R is still 0.7 (the ball stayed at A's commit).

FACT B (fresh, registered): NAME = "QUORINA" (selection rule: the first of
[QUORINA, MERIDIA, KORVETH, XYRANNE] with 0 occurrences in train AND val; 7
chars like ZEPHYRA so PRE=130/POST=119 hold verbatim; onset char Q train
count 230, Z's own rarity class — 161). HOSTS = ["CAMILLO", "AUTOLYCUS"] (72
+ 67 occurrences; the same Winter's Tale neighborhood as A's FLORIZEL/
ELIZABETH so the context style is comparable; fresh strings => the window
sets are disjoint from A's by construction, gated). SPLICE RNG 24302,
install generator seed 24314 (both free of every registered block; e043's
own draws are 24301/24303-24313/24321/24331).

MACHINERY: lab/g1b_continuity.py VERBATIM via import (which owns the
G1.G1_CFG/G1_PARAMS -> 2.74M patch), lab/g1_anchored_ball.py for
CommittedGPT / g1_wash / evl_load / instruments, e043's Dmix stream
composition VERBATIM in g1's phase-0a operationalization (16 spliced install
windows + 16 paired originals + 32 random corpus per step, draw order
ix(16)/aj(16)/rj(32), full-token CE over the union, AdamW (0.9,0.95) wd 0.1
clip 1.0, house cosine warmup 100; g1's registered DmixCorpus adaptation —
the same install that built A's root), dose s300 per the critic's spec.
Free-run battery: e043's R3 convention (4 install + 4 held 120-char prompts,
350 tokens, temp 0.8 top-k 40, per-prompt-index torch seeds, name-count
regexes per 10k chars); clean-judge score via e110's machinery IF importable
(it is, but e110 patches CUDA off at module level -> imported LAZILY, after
all training; judge = clean full recompute CE of each completion's tail-128
window under (i) the root and (ii) the reference leg's final net).

COMPUTE ENVELOPE (hard): 4 trainings (2x 50-step wash, 2x 300-step install),
each <= 180 s GPU / 1800 s CPU; cooldown 120 s before each training; gpu_ok()
double-poll before every launch, WAIT up to 10 min for free windows (g2f
serialization), CPU only if busy > 10 min (documented); mid-run contention
poll every 25 steps with in-place migration; evals/batteries CPU; NO
concurrent GPU. 2,739,072 params (inside the <=100M free tier; the stated
reason is g1b's own: CONTINUITY on the e131 line).

Outputs: runs/g1bW/{metrics.json, museum_test.png}; checkpoints
runs/checkpoints/g1bW_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit + push (the coordinator folds).

Run:  cd lab && python g1bW_second_fact.py    (G1BW_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import os
import random
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G1BW_SMOKE") == "1"
if SMOKE:
    os.environ["G1B_SMOKE"] = "1"     # cascades to G1_SMOKE inside g1b's import

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

torch.set_num_threads(8)                              # e152R/e143/e184/e179

import common                                          # noqa: E402
from common import CharCorpus, cooldown, gpu_ok,     # noqa: E402
                      gpu_status, run_dir, save_json, set_seed
import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
                                                      # (owns the 2.74M patch
                                                      # + GB.CKPT_DIR /
                                                      # GB.ROOT_CK verbatim)
import g1_anchored_ball as G1                          # noqa: E402 — the
                                                      # machinery (patched to
                                                      # 2.74M by g1b's import)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

# ======================================================================
# THE CONFIG (registered; the whole delta from g1bR's W1 cell)
# ======================================================================
WASH_SEED = 10907        # THE lineage seed (g1bR's first replicate)
R_CLAIM = 0.7            # the wall's radius (g1b/g1bR's W1, VERBATIM)
WASH_STEPS_CKS = (1, 2, 4, 10, 50) if not SMOKE else (1, 2, 4)
INSTALL_STEPS = 300 if not SMOKE else 8
INSTALL_CKS = (1, 2, 4, 10, 50, 100, 200, 300) if not SMOKE else (1, 2, 4, 8)

B_NAME = "QUORINA"       # registered selection rule (docstring)
B_HOSTS = ["CAMILLO", "AUTOLYCUS"]
B_SPLICE_RNG = 24302     # fresh (e043's own SPLICE_RNG is 24301)
B_INSTALL_SEED = 24314   # fresh (e043's GEN_D ladder ends at 24313)

B_INSTALL_BAR = 0.7      # the critic's own threshold ("B installs (>= 0.7)")
RIDER_ROWS = (0,) + tuple(range(121, 130))   # the critic's collapse channels
TRAIN_CAP_GPU, TRAIN_CAP_CPU = 180.0, 1800.0
COOLDOWN_S = 120.0       # the mission's envelope (overrides g1's 60)
GPU_WAIT_MAX = 600.0     # wait up to 10 min for a free GPU window (mission)

RIDER_CK = GB.CKPT_DIR / "g1bR_W1_10907_s300.pt"
CKPT_DIR = GB.CKPT_DIR

G1BW_BARS = {
    "SPLINT-REFUTED": ("fires if B installs (B-ruler >= 0.7) with A >= 0.5 — "
                       "a real memory architecture; the wall is not a "
                       "splint."),
    "MUSEUM": ("fires if B fails (<= 0.27) at healthy CE — the wall is a "
               "splint; rescope the claim to 'the wall freezes the organism; "
               "memory survives freezing'."),
    "ZERO-SUM": ("fires if B installs and A dies (A < 0.5) — displacement "
                 "inside the ball kills; the ball is a single-exhibit museum "
                 "by geometry."),
    "channel_rider": ("A's wpe-band channel and row0 strength at every "
                      "checkpoint (co-reported; scopes which channels of A "
                      "the wall protected during B's install)."),
    "operationalizations": {
        "A_ruler": "g1b fact battery, g-12 form (mean p(Z), install-60 "
                   "contexts, offset -12); A read at the END of B's install",
        "B_ruler": "same battery form for B (mean p(Q), B install-60 "
                   "contexts, offset -12); read at the END",
        "B_installs": "B-ruler >= 0.7 (the critic's threshold)",
        "A_holds": "A-ruler >= 0.5 (g1's MAINTAIN_BAR)",
        "healthy_CE": "CE_r <= root CE_r + 0.30 (g1's CE_NOISE_BAR, the "
                      "arc's established walled-arm health bound), at the END",
        "ambiguous": "B strictly in (0.27, 0.7), or B fails at UNhealthy CE "
                     "-> NO bar fires; reported as AMBIGUOUS, no re-tuning",
    },
}

deviations: list[str] = [
    "MISSION ENVELOPE overrides two g1 policies (registered): cooldown 120 s "
    "before each training (g1's band was 60-120), and the GPU gate WAITS up "
    "to 10 min for a free window (g2f serialization; g1's pick_dev parked on "
    "first failure) — fresh gate per training, CPU only if busy > 10 min.",
    "WASH PHASE re-run (50 steps, seed 10907) instead of continuing from a "
    "stored +50 checkpoint: g1bR saved only each arm's +300 final, and the "
    "+50 state is bit-reproducible from the recorded seed (G-WASHREP gates "
    "the reproduction against g1bR's stored +50 reading).",
    "FRESH OPTIMIZER for the install phase (AdamW state reset at the wash/"
    "install boundary): the install is a NEW task in the arc's own convention "
    "(e043's installs each start a fresh optimizer); the WALL (anchor + R) is "
    "the continuous object, not the optimizer.",
    "B INSTALL DOSE = 300 steps (the critic's spec 'e043-Dmix install "
    "VERBATIM, 300 steps'); A's own root install ran s400 (g1's phase 0a). "
    "Stream composition, draw order, weighting (full-token union CE), "
    "optimizer and schedule are g1's phase-0a operationalization VERBATIM — "
    "the exact machinery that built A's root.",
    "CHECKPOINT SAVING trimmed to the states the fold needs: walled post-wash "
    "+50, both legs' install finals (g1b's wash-arm convention; intermediate "
    "states are bit-reproducible from the recorded seeds).",
    "E110 JUDGE IMPORT IS DELAYED to after all training: e110_field_floor.py "
    "sets CUDA_VISIBLE_DEVICES=-1 and patches torch.cuda off at module level "
    "(CPU-only experiment); importing it lazily keeps the GPU envelope intact "
    "and satisfies 'if the e110 judge machinery is importable'.",
    "E110 JUDGE WINDOW ADAPTED to block-256 nets: e110's own rig is ctx512 "
    "(tail 448..511 of 512); here each completion stream is cropped to its "
    "last 256 positions and the judge scores the tail-128 window (the same "
    "clean full-recompute judge_windows arithmetic, e110 VERBATIM inside the "
    "crop). Two judges co-reported: the ROOT (knows A) and the reference "
    "leg's final net (the best available B-knower).",
    "Smoke mode trims: 4-step wash, 8-step install, 60-token free-runs, "
    "2 prompts, no cooldowns — nothing adjudicated.",
]

CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1bW", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ GPU gate
def wait_gpu(tag: str) -> torch.device:
    """The mission's envelope: double-poll; WAIT for free windows (g2f's
    trainings are short bursts with cooldowns); CPU only if busy > 10 min.
    Fresh gate per training (a registered deviation from g1's park-once —
    under fleet serialization re-checking is the correct semantics)."""
    G1.GPU_PARKED, G1.PARK_REASON = False, None
    if not torch.cuda.is_available():
        G1.GPU_PARKED, G1.PARK_REASON = True, "no CUDA"
        return CPU
    t0, waited, s1 = time.time(), 0.0, gpu_status()
    while (time.time() - t0) <= GPU_WAIT_MAX:
        if gpu_ok():
            time.sleep(5)
            if gpu_ok():
                s2 = gpu_status()
                log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                    f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                    f"{s2['mem_total']:.0f}MB"
                    + (f"; waited {waited:.0f}s for the window"
                       if waited > 0 else "") + ")")
                return torch.device("cuda")
        time.sleep(30)
        waited = time.time() - t0
    G1.GPU_PARKED = True
    G1.PARK_REASON = f"busy >{GPU_WAIT_MAX:.0f}s (waited from {s1})"
    log(f"[gpu] '{tag}' and this training run CPU (GPU busy >"
        f"{GPU_WAIT_MAX:.0f}s: {s1})")
    return CPU


# ------------------------------------------------------------------ B install
def install_b(tag: str, net0, theta0_flat: torch.Tensor, inst_x, anchor_b,
              train_ids, eval_fn, ckpt_steps=INSTALL_CKS, steps=INSTALL_STEPS,
              seed=B_INSTALL_SEED):
    """B's install (e043-Dmix VERBATIM in g1's phase-0a operationalization):
    per step ix(16) spliced B-windows + aj(16) paired originals + rj(32)
    random corpus; full-token CE over the union; AdamW (0.9,0.95) wd 0.1,
    lr 1e-3 house cosine (warmup 100), clip 1.0; the wall (if committed)
    projects at every forward — the anchor NEVER moves. Checkpoint snapshots
    + CPU evals at ckpt_steps; displacement vs theta0 (the anchor origin)
    measured per step; per-step input md5 recorded."""
    dev = wait_gpu(tag)
    cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    ckpt_set = set(ckpt_steps)
    n_inst, n_anc = inst_x.shape[0], anchor_b.shape[0]
    traj, sds, cells = [], {}, {}
    x_hashes: dict[int, str] = {}
    prev = theta0_flat.clone()
    t_start = time.time()
    step = 0
    for step in range(1, steps + 1):
        f = common.cosine_lr(step - 1, steps)       # house schedule, warmup 100
        for g in opt.param_groups:
            g["lr"] = 1e-3 * f
        ix = torch.randint(n_inst, (G1.NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (G1.CORP_BS - G1.MIX_RANDOM,), generator=gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.MIX_RANDOM,),
                           generator=gen)
        corp = torch.cat([anchor_b[aj],
                          torch.stack([train_ids[s: s + G1.BLOCK]
                                       for s in rj])], 0)
        nw = inst_x[ix]
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        xd, yd = x.to(dev), y.to(dev)
        logits, _ = net(xd)                    # <- the wall projects here
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               yd.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        cur = G1.flat_params_cpu(net)
        cum_disp = float(torch.norm(cur - theta0_flat))
        row = {"step": step, "lr": 1e-3 * f, "ce_batch": float(loss.item()),
               "cum_disp": cum_disp,
               "step_disp": float(torch.norm(cur - prev)),
               "elapsed_s": round(time.time() - t_start, 1)}
        prev = cur
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            cells[step] = eval_fn(sd_cpu, f"{tag}_i{step}")
            row["d_proj"] = (min(cum_disp, net0.R)
                             if getattr(net0, "R", None) else None)
            log(f"  [{tag}] CKPT +{step:4d} A {cells[step]['A_gm12']:.4f} "
                f"B {cells[step]['B_gm12']:.4f} CE_R {cells[step]['ce_r']:.4f} "
                f"|d| {cum_disp:.4f} (CE {float(loss.item()):.4f})")
        traj.append(row)
        if step % 50 == 0 and step not in ckpt_set:
            log(f"  [{tag}] s{step:4d} CE {float(loss.item()):.4f} "
                f"|d| {cum_disp:.4f} ({row['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            G1.trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % G1.MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                G1.device_events.append({"tag": tag, "step": step,
                                         "event": "MID-RUN MIGRATION",
                                         "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> CPU")
                G1.migrate_to_cpu(net, opt)
                dev = CPU
    net.eval()
    return {"sds": sds, "cells": cells, "traj": traj, "steps_ran": step,
            "seed": seed, "steps": steps, "x_hashes": x_hashes,
            "device": str(dev),
            "runtime_s": round(time.time() - t_start, 1)}


# ------------------------------------------------------------------ free-run
def freerun_battery(net, corpus, prompts_a, prompts_b, tag):
    """e043's R3 convention: per prompt, torch.manual_seed(prompt index),
    350 tokens, temp 0.8 top-k 40; count A-name/B-name occurrences + stray
    Z-words per 10k continuation chars; keep completions verbatim."""
    import re as _re
    common.DEVICE = "cpu"                    # eval nets live on CPU
    out = {"tag": tag, "A": {}, "B": {}}
    for fact, prompts, name in (("A", prompts_a, G1.NAME),
                                ("B", prompts_b, B_NAME)):
        n_toks = 60 if SMOKE else 350
        counts = {"name": 0, "other_name": 0, "stray_Z_words": 0,
                  "chars": 0}
        comps = []
        for i, pr in enumerate(prompts):
            torch.manual_seed(i)             # e043 R3 seed convention
            gen_txt = common.generate(net, corpus, pr, max_new_tokens=n_toks,
                                      temperature=0.8, top_k=40)
            cont = gen_txt[len(pr):]
            other = B_NAME if fact == "A" else G1.NAME
            counts["name"] += len(_re.findall(
                r"(?<![A-Za-z])" + _re.escape(name) + r"(?![A-Za-z])", cont))
            counts["other_name"] += len(_re.findall(
                r"(?<![A-Za-z])" + _re.escape(other) + r"(?![A-Za-z])", cont))
            counts["stray_Z_words"] += len(_re.findall(
                r"(?<![A-Za-z])Z[A-Za-z]*", cont))
            counts["chars"] += len(cont)
            comps.append({"prompt": pr, "continuation": cont})
        per10k = {k: (round(v / max(counts["chars"], 1) * 1e4, 2)
                      if counts["chars"] else None)
                  for k, v in counts.items() if k != "chars"}
        out[fact] = {"counts": {k: v for k, v in counts.items()
                                if k != "chars"},
                     "chars": counts["chars"], "per_10k_chars": per10k,
                     "completions": comps}
        log(f"[freerun:{tag}/{fact}] {name} x{counts['name']} "
            f"({per10k['name']}/10k) | stray-Z {counts['stray_Z_words']} "
            f"| {counts['chars']} chars")
    return out


def judge_battery(stream_pairs, judges):
    """e110's clean-judge machinery, imported LAZILY (it patches CUDA off).
    stream_pairs: [(label, prompt, continuation)]; judges: {name: net}."""
    try:
        import e110_field_floor as E110                     # noqa: F401
    except Exception as e:                                   # noqa: BLE001
        return {"importable": False, "error": repr(e)}
    out = {"importable": True,
           "note": "e110 judge_windows VERBATIM; stream cropped to last 256 "
                   "positions (block-256 nets; e110's own rig is ctx512), "
                   "judged window = tail 128 transitions"}
    for jname, jnet in judges.items():
        out[jname] = {}
        for label, prompt, cont in stream_pairs:
            stream = prompt + cont
            idx = torch.tensor([common.CharCorpus and 0], dtype=torch.long) \
                if False else None
            # encode via the judge's own corpus (the shared char vocab)
            idx = _encode(stream)
            idx_j = idx[-256:].unsqueeze(0)
            T = idx_j.shape[1]
            lo, hi = T - 128, T - 1
            lg = E110.manual_all_logits(jnet, idx_j)
            sc = float(E110.judge_windows(lg, idx_j, [(lo, hi)])[(lo, hi)][0])
            out[jname][label] = sc
        out[jname]["mean"] = float(np.mean(list(out[jname].values())))
    return out


_CORPUS_GLOBAL = None


def _encode(s: str) -> torch.Tensor:
    return _CORPUS_GLOBAL.encode(s)


# ------------------------------------------------------------------ main

def main():
    global _CORPUS_GLOBAL
    rd = run_dir("g1bW_smoke" if SMOKE else "g1bW")
    log(f"G1BW THE WALL'S MUSEUM TEST (smoke={SMOKE}) -> {rd}")
    set_seed(G1.INSTALL_SEED)

    if not RIDER_CK.exists():
        raise RuntimeError(f"rider checkpoint missing: {RIDER_CK} "
                           "(spec says report and skip — but it EXISTS in "
                           "this repo; refusing to run without it would be "
                           "bar shopping the rider)")

    # ---------------- protocol rebuild, A side (g1bR's main VERBATIM) -------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    _CORPUS_GLOBAL = corpus
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    bid = stoi[B_NAME[0]]                    # the B-ruler's read token ('Q')
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
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
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"A protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG "
        f"{E43.SPLICE_RNG})")
    name_ids = corpus.encode(G1.NAME)
    b_name_ids = corpus.encode(B_NAME)

    # A measurement pool (e152's locked j=54 windows, instrument only)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - G1.PRE - G1.RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + G1.SITE_CONT]
        w = torch.cat([pre, name_ids, post])
        if len(w) != G1.BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {G1.BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)

    # ---------------- the neutral stream (e170 VERBATIM via g1bR) ----------
    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK]
                                  for s in n_starts])
    log(f"neutral bank 16x{G1.BLOCK} (seed {G1.E170_ANCHOR_SEED}, "
        f"{rejections} rejections/{tries} tries)")

    # ---------------- A batteries (e119/e176n verbatim) ---------------------
    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    # ---------------- B construction (fresh host windows + fresh name) -----
    for c in B_NAME:
        assert c in stoi, f"B name char {c} not in vocab"
    b_train_occ = len(E43.find_occ(train_text, B_NAME))
    b_val_occ = len(E43.find_occ(val_text, B_NAME))
    host_occ_b = []
    for host in B_HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ_b.append((p, host))
    brng = random.Random(B_SPLICE_RNG)
    brng.shuffle(host_occ_b)
    install_occ_b, held_occ_b = host_occ_b[:60], host_occ_b[60:90]
    b_mix = {h: sum(1 for _, hh in install_occ_b if hh == h) for h in B_HOSTS}
    a_positions = {p for p, _ in install_occ} | {p for p, _ in held_occ}
    b_positions = {p for p, _ in install_occ_b} | {p for p, _ in held_occ_b}
    overlap = sorted(a_positions & b_positions)

    # B install windows (e043 build_win VERBATIM form; PRE=130, POST=119 —
    # B_NAME is 7 chars like ZEPHYRA, so the geometry is IDENTICAL)
    assert len(B_NAME) == len(G1.NAME), "B name length must match A's (7)"

    def build_win_b(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], b_name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    inst_b = torch.stack([build_win_b(p, h) for p, h in install_occ_b])
    anchor_b = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                            for p, _ in install_occ_b])
    # B batteries: SAME form as A's (contexts ending at B's host onsets)
    bbat_ids, bheld_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ_b]
        bbat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ_b]
        bheld_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    log(f"B protocol built: {B_NAME} (train {b_train_occ}/val {b_val_occ} "
        f"occ), hosts {b_mix} install60/held30 (SPLICE_RNG {B_SPLICE_RNG})")

    # ---------------- the per-checkpoint eval dial (A + B + CE + channels) --
    def eval_dial(sd: dict, tag: str) -> dict:
        """evl_load (settle+disarm, g1's PIVOT) -> A/B batteries + CE_R +
        site read + the channel rider census (row0 + the 121-129 band, on
        A's g0 readout = g1b's census_old readout VERBATIM)."""
        net = G1.evl_load(sd)
        out = {"tag": tag}
        for j in G1.GEOS:
            out[f"A_g{j:+d}" if False else f"A_geo{j}"] = None
        out["A_cells"] = {j: G1.battery_cell(net, bat_ids[j], zid)
                          for j in G1.GEOS}
        out["A_held"] = {j: G1.battery_cell(net, held_ids[j], zid)
                         for j in G1.GEOS}
        out["B_cells"] = {j: G1.battery_cell(net, bbat_ids[j], bid)
                          for j in G1.GEOS}
        out["B_held"] = {j: G1.battery_cell(net, bheld_ids[j], bid)
                         for j in G1.GEOS}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        out["site_read"] = G1.read_fact_at(net, pool_x, name_ids, zid,
                                           G1.SITE_ADDR_ROW, G1.SITE_Z_XCOL)
        cen = G1.row_census(net, RIDER_ROWS,
                            lambda n: G1.battery_pz(n, bat_ids[0], zid))
        co = cen["rows"]
        out["census"] = {"base_readout": cen["base_readout"],
                         "row0_strength": co["0"]["strength"],
                         "A129": co["129"]["strength"],
                         "band121_129_max": max(co[str(r)]["strength"]
                                                for r in range(121, 130)),
                         "band_rows": {str(r): {"strength": co[str(r)]["strength"],
                                                "content": co[str(r)]["content"]}
                                       for r in range(121, 130)}}
        flat = {"A_gm12": out["A_cells"][-12]["mean_pz"],
                "A_g0": out["A_cells"][0]["mean_pz"],
                "A_gp12": out["A_cells"][12]["mean_pz"],
                "A_held30_gm12": out["A_held"][-12]["mean_pz"],
                "B_gm12": out["B_cells"][-12]["mean_pz"],
                "B_g0": out["B_cells"][0]["mean_pz"],
                "B_gp12": out["B_cells"][12]["mean_pz"],
                "B_held30_gm12": out["B_held"][-12]["mean_pz"],
                "ce_r": out["ce_r"],
                "site_read_onset": out["site_read"]["pz_onset_mean"],
                "site_read_span": out["site_read"]["pname_mean_over7"],
                "row0_strength": out["census"]["row0_strength"],
                "A129": out["census"]["A129"],
                "band121_129_max": out["census"]["band121_129_max"]}
        out["flat"] = flat
        log(f"[{tag}] A g-12 {flat['A_gm12']:.4f} g0 {flat['A_g0']:.4f} | "
            f"B g-12 {flat['B_gm12']:.4f} g0 {flat['B_g0']:.4f} | CE_R "
            f"{flat['ce_r']:.4f} | row0 S {flat['row0_strength']:+.4f} | "
            f"band(max) {flat['band121_129_max']:+.4f}")
        del net
        return flat

    def anchor_check(sd: dict) -> dict:
        """G-ANCHOR: the ball's anchor is still bit-equal theta0 at R=0.7."""
        body, anch = G1.split_anchored_sd(sd)
        ok_R = bool(torch.equal(anch.get("anch__R"),
                                torch.tensor(R_CLAIM)))
        bad = [k for k, v in anch.items()
               if k != "anch__R" and not torch.equal(
                   v, theta0[k.replace("anch__", "").replace("_", ".")])]
        # note: sd keys use '.' in param names; anchors replace '.' with '_'
        bad = []
        for k, v in anch.items():
            if k == "anch__R":
                continue
            body_key = _anch_to_body(k)
            if body_key not in theta0 or not torch.equal(v, theta0[body_key]):
                bad.append(k)
        return {"R": float(anch["anch__R"]) if "anch__R" in anch else None,
                "n_anchor_tensors": len(anch) - 1,
                "mismatched": bad,
                "pass": bool(ok_R and not bad)}

    def _anch_to_body(k: str) -> str:
        # invert CommittedGPT.commit's name mangling: the FIRST '.'-free
        # reconstruction is ambiguous, so we map via the root net's own
        # named_parameters (deterministic order).
        return _ANCH_MAP.get(k, k)

    # =====================================================================
    # THE ROOT (the arc's consolidated root, loaded DIRECTLY; bit-exact)
    # =====================================================================
    log("=" * 78)
    root_net = G1.load_g1(CKPT_DIR / GB.ROOT_CK)
    assert root_net.num_params() == GB.G1B_PARAMS
    theta0 = {k: v.detach().clone() for k, v in root_net.state_dict().items()}
    raw = torch.load(CKPT_DIR / GB.ROOT_CK, map_location="cpu",
                     weights_only=False)
    raw_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    md0 = max(float((theta0[k].float() - raw_sd[k].float()).abs().max())
              for k in raw_sd)
    G_BITEXACT = {"checkpoint": f"runs/checkpoints/{GB.ROOT_CK}",
                  "n_tensors": len(raw_sd),
                  "max_abs_diff_vs_file": md0, "pass": bool(md0 == 0.0)}
    assert G_BITEXACT["pass"], f"root load not bit-exact: {md0}"
    theta0_flat = G1.flat_params_cpu(root_net)
    _ANCH_MAP = {}
    for name, _p in root_net.named_parameters():
        _ANCH_MAP["anch__" + name.replace(".", "_")] = name

    root_cells = eval_dial(theta0, "root")
    G_ROOT = {"bar": G1.EXPRESS_BAR, "gm12": root_cells["A_gm12"],
              "pass": bool(root_cells["A_gm12"] >= G1.EXPRESS_BAR)}
    log(f"G-ROOT: root A g-12 {root_cells['A_gm12']:.4f} "
        f"(bar >= {G1.EXPRESS_BAR}): "
        f"{'PASS' if G_ROOT['pass'] else 'FAIL'}")
    root_ce_r = root_cells["ce_r"]

    # G-BFRESH
    G_BFRESH = {
        "B_name": B_NAME, "train_occ": b_train_occ, "val_occ": b_val_occ,
        "n_install": len(install_occ_b), "n_held": len(held_occ_b),
        "host_mix": b_mix,
        "window_overlap_with_A": overlap,
        "B_battery_base_gm12": root_cells["B_gm12"],
        "B_battery_base_g0": root_cells["B_g0"],
        "pass": bool(b_train_occ == 0 and b_val_occ == 0
                     and len(install_occ_b) == 60 and len(held_occ_b) == 30
                     and not overlap and root_cells["B_gm12"] <= 0.05
                     and root_cells["B_g0"] <= 0.05),
    }
    assert G_BFRESH["pass"], f"G-BFRESH FAILED: {G_BFRESH}"
    log(f"G-BFRESH: {B_NAME} 0 occ, hosts {b_mix}, 0 window overlap with A, "
        f"B base g-12 {root_cells['B_gm12']:.4f} <= 0.05: PASS")

    # =====================================================================
    # THE WALLED LEG — commit(0.7) + 50 wash + B install UNDER the projection
    # =====================================================================
    log("=" * 78)
    log(f"WALLED LEG — commit({R_CLAIM}) -> {WASH_STEPS_CKS[-1]} wash steps "
        f"(seed {WASH_SEED}) -> {B_NAME} install {INSTALL_STEPS} steps UNDER "
        f"the projection")
    if not SMOKE:
        cooldown(COOLDOWN_S)
    net0 = G1.CommittedGPT(GB.G1B_CFG)
    net0.load_state_dict(theta0)
    net0.commit(R_CLAIM)
    body, _ = G1.split_anchored_sd(net0.state_dict())
    md = max(float((body[k].float() - theta0[k].float()).abs().max())
             for k in theta0)
    anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                  for n, p in net0.named_parameters())
    G_BITROOT = {"max_abs_diff": md, "anchors_bit_equal": bool(anch_ok),
                 "n_anchor_tensors": net0._n_anchor_tensors,
                 "pass": bool(md == 0.0 and anch_ok)}
    assert G_BITROOT["pass"], "wall root != theta0"
    log(f"G-BITROOT: max|diff| {md:.1e}, anchors bit-equal: PASS")

    w_wash = G1.g1_wash("W1w", net0, anchor_neutral, train_ids, itos,
                        r_eval_xy, gm12_ids, g0_ids, zid, target_mode="true",
                        noise_seed=0, ckpt_steps=WASH_STEPS_CKS,
                        seed=WASH_SEED)
    w_wash_gm12 = {t["step"]: t["g_m12_mean_pz"] for t in w_wash["traj"]
                   if "g_m12_mean_pz" in t}
    w_wash_disp = {t["step"]: t["cum_disp"] for t in w_wash["traj"]
                   if "g_m12_light" in t or "g_m12_mean_pz" in t}
    wash50_sd = w_wash["sds"][WASH_STEPS_CKS[-1]]

    # G-WASHREP + G-CTRL use g1bR's stored seed-10907 readings
    g1bR_ref = {"W1_plus50_gm12": 0.915731336593628,   # runs/g1bR traces
                "C_plus50_gm12": 0.0027922233065366517,
                "C_plus2_gm12": 0.4611}
    G_WASHREP = {"bar": 0.05, "mine": w_wash_gm12.get(WASH_STEPS_CKS[-1]),
                 "g1bR": g1bR_ref["W1_plus50_gm12"],
                 "pass": bool(w_wash_gm12.get(WASH_STEPS_CKS[-1]) is not None
                              and abs(w_wash_gm12[WASH_STEPS_CKS[-1]]
                                      - g1bR_ref["W1_plus50_gm12"]) <= 0.05)}
    log(f"G-WASHREP: walled wash +50 g-12 {w_wash_gm12.get(50)} vs g1bR "
        f"{g1bR_ref['W1_plus50_gm12']:.4f} (tol 0.05): "
        f"{'PASS' if G_WASHREP['pass'] else 'FAIL'}")
    save_ckpt("g1bW_W1_wash50", wash50_sd,
              {"desc": f"e131 root + commit({R_CLAIM}) + 50-step neutral wash "
                       f"(seed {WASH_SEED}) — the walled pre-install state",
               "steps": int(WASH_STEPS_CKS[-1]), "R": R_CLAIM,
               "wash_seed": WASH_SEED,
               "base": f"runs/checkpoints/{GB.ROOT_CK}"})

    # reload the washed state ARMED (the wall continues; anchor = theta0)
    net_w = G1.CommittedGPT(GB.G1B_CFG)
    body_w, anch_w = G1.split_anchored_sd(wash50_sd)
    net_w.load_state_dict(body_w)
    G1._restore_anchors(net_w, anch_w)
    assert net_w.anchored and net_w.R == R_CLAIM
    ac0 = anchor_check(net_w.state_dict())
    assert ac0["pass"], f"anchor drifted before install: {ac0}"
    log(f"G-ANCHOR[pre-install]: R={ac0['R']}, {ac0['n_anchor_tensors']} "
        f"anchor tensors still bit-equal theta0: PASS")

    if not SMOKE:
        cooldown(COOLDOWN_S)
    w_g_anchors = {"pre_install": ac0}

    def eval_w(sd, tag):
        c = eval_dial(sd, tag)
        w_g_anchors[tag] = anchor_check(sd)
        return c

    w_ins = install_b("W1w_install", net_w, theta0_flat, inst_b, anchor_b,
                      train_ids, eval_w)
    for tag, ac in w_g_anchors.items():
        assert ac["pass"] or SMOKE, f"G-ANCHOR FAILED at {tag}: {ac}"
    G_ANCHOR = {"per_checkpoint": {k: {"R": v["R"], "pass": v["pass"],
                                       "mismatched": v["mismatched"]}
                                   for k, v in w_g_anchors.items()},
                "pass": bool(all(v["pass"] for v in w_g_anchors.values()))}
    log(f"G-ANCHOR: the ball stayed at A's commit at every install "
        f"checkpoint: {'PASS' if G_ANCHOR['pass'] else 'FAIL'}")
    w_final_sd = w_ins["sds"][max(w_ins["sds"])]
    save_ckpt("g1bW_W1_install300", w_final_sd,
              {"desc": f"walled leg final: commit({R_CLAIM}) + 50 wash "
                       f"(seed {WASH_SEED}) + {B_NAME} Dmix install "
                       f"s{w_ins['steps_ran']} UNDER the projection "
                       f"(gen seed {B_INSTALL_SEED})",
               "steps": int(w_ins["steps_ran"]), "R": R_CLAIM,
               "wash_seed": WASH_SEED, "install_seed": B_INSTALL_SEED,
               "base": f"runs/checkpoints/{GB.ROOT_CK}"})

    # =====================================================================
    # THE REFERENCE LEG — unwalled washed control at +50, same B install
    # =====================================================================
    log("=" * 78)
    log(f"REFERENCE LEG (co-reported, not gating) — UNWALLED + "
        f"{WASH_STEPS_CKS[-1]} wash steps (seed {WASH_SEED}) -> the SAME "
        f"{B_NAME} install, no projection")
    if not SMOKE:
        cooldown(COOLDOWN_S)
    netR0 = G1.evl_load(theta0)               # uncommitted (plain TinyGPT)
    r_wash = G1.g1_wash("Cref", netR0, anchor_neutral, train_ids, itos,
                        r_eval_xy, gm12_ids, g0_ids, zid, target_mode="true",
                        noise_seed=0, ckpt_steps=WASH_STEPS_CKS,
                        seed=WASH_SEED)
    r_wash_gm12 = {t["step"]: t["g_m12_mean_pz"] for t in r_wash["traj"]
                   if "g_m12_mean_pz" in t}
    G_CTRL = {"bar": G1.SHUT_BAR,
              "gm12_at_50": r_wash_gm12.get(WASH_STEPS_CKS[-1]),
              "pass": bool(r_wash_gm12.get(WASH_STEPS_CKS[-1], 1.0)
                           <= G1.SHUT_BAR)}
    log(f"G-CTRL: reference wash +50 A g-12 {r_wash_gm12.get(50)} "
        f"(bar <= {G1.SHUT_BAR}): "
        f"{'PASS' if G_CTRL['pass'] else 'FAIL'}")
    net_r = G1.evl_load(r_wash["sds"][WASH_STEPS_CKS[-1]])   # uncommitted
    if not SMOKE:
        cooldown(COOLDOWN_S)
    r_ins = install_b("Cref_install", net_r, theta0_flat, inst_b, anchor_b,
                      train_ids, lambda sd, tag: eval_dial(sd, tag))
    r_final_sd = r_ins["sds"][max(r_ins["sds"])]
    save_ckpt("g1bW_REF_install300", r_final_sd,
              {"desc": f"reference leg final: unwashed-commit root + 50-step "
                       f"neutral wash (seed {WASH_SEED}) + {B_NAME} Dmix "
                       f"install s{r_ins['steps_ran']} (gen seed "
                       f"{B_INSTALL_SEED}) — the capacity reference",
               "steps": int(r_ins["steps_ran"]), "R": None,
               "wash_seed": WASH_SEED, "install_seed": B_INSTALL_SEED,
               "base": f"runs/checkpoints/{GB.ROOT_CK}"})

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES (pin + inputs)
    # =====================================================================
    G_PIN = {"R": R_CLAIM, "bound": R_CLAIM + G1.PIN_FUZZ_BAR,
             "per_ckpt": {}, "pass": None}
    pin_pts = dict(w_wash_disp)
    for s, row in enumerate(w_ins["traj"]):
        if (s + 1) in w_ins["cells"]:
            pin_pts[10000 + (s + 1)] = row["cum_disp"]   # install phase
    G_PIN["per_ckpt"] = {"wash": w_wash_disp,
                         "install": {str(s): t["cum_disp"]
                                     for s, t in enumerate(w_ins["traj"])
                                     if (s + 1) in w_ins["cells"]}}
    mx = max(list(w_wash_disp.values())
             + [t["cum_disp"] for s, t in enumerate(w_ins["traj"])
                if (s + 1) in w_ins["cells"]])
    G_PIN["max_raw_disp"] = mx
    G_PIN["pass"] = bool(mx <= R_CLAIM + G1.PIN_FUZZ_BAR)
    log(f"G-PIN: max raw |d| vs theta0 over wash+install ckpts {mx:.4f} <= "
        f"{R_CLAIM + G1.PIN_FUZZ_BAR:.2f}: "
        f"{'PASS' if G_PIN['pass'] else 'FAIL'}")

    shared_steps = sorted(set(w_ins["x_hashes"]) & set(r_ins["x_hashes"]))
    G_INPUTS = {"steps_compared": len(shared_steps),
                "identical_steps": sum(
                    1 for s in shared_steps
                    if w_ins["x_hashes"][s] == r_ins["x_hashes"][s]),
                "pass": bool(shared_steps
                             and all(w_ins["x_hashes"][s] == r_ins["x_hashes"][s]
                                     for s in shared_steps))}
    assert G_INPUTS["pass"], f"G-INPUTS FAILED: {G_INPUTS}"
    log(f"G-INPUTS: install batches bit-identical across legs "
        f"({G_INPUTS['identical_steps']}/{G_INPUTS['steps_compared']} steps, "
        f"md5): PASS — the wall is the only delta")

    gates = {"G_BITEXACT": G_BITEXACT, "G_ROOT": G_ROOT,
             "G_BITROOT": G_BITROOT, "G_BFRESH": G_BFRESH,
             "G_CTRL": G_CTRL, "G_WASHREP": G_WASHREP, "G_PIN": G_PIN,
             "G_INPUTS": G_INPUTS, "G_ANCHOR": G_ANCHOR}
    gates_pass = bool(all(g["pass"] for g in gates.values()))

    # =====================================================================
    # ADJUDICATION (the critic's VERBATIM bars; operationalized as registered)
    # =====================================================================
    last_ck = max(w_ins["cells"])
    w_fin = w_ins["cells"][last_ck]
    r_fin = r_ins["cells"][max(r_ins["cells"])]
    A_final, B_final, CE_final = w_fin["A_gm12"], w_fin["B_gm12"], w_fin["ce_r"]
    B_installs = bool(B_final >= B_INSTALL_BAR)
    A_holds = bool(A_final >= G1.MAINTAIN_BAR)
    ce_healthy = bool(CE_final <= root_ce_r + G1.CE_NOISE_BAR)
    A_min = min(c["A_gm12"] for c in w_ins["cells"].values())

    SPLINT_REFUTED = bool(B_installs and A_holds)
    MUSEUM = bool(B_final <= G1.SHUT_BAR and ce_healthy)
    ZERO_SUM = bool(B_installs and not A_holds)

    if not gates_pass:
        failed = [k for k, g in gates.items() if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a construction gate failed — bars reported as measured but "
                  f"nothing adjudicated; failed: {failed}")
    elif SPLINT_REFUTED:
        verdict = "SPLINT-REFUTED"
        clause = (f"B installs (B-ruler {B_final:.4f} >= {B_INSTALL_BAR}) with "
                  f"A held (A-ruler {A_final:.4f} >= {G1.MAINTAIN_BAR}; min "
                  f"over install checkpoints {A_min:.4f}) at CE_r "
                  f"{CE_final:.4f} (root {root_ce_r:.4f} + 0.30 bar "
                  f"{root_ce_r + G1.CE_NOISE_BAR:.4f}) — a real memory "
                  "architecture; the wall is not a splint: the FIRST "
                  "genuinely sequential memory in the lab's history.")
    elif MUSEUM:
        verdict = "MUSEUM"
        clause = (f"B fails (B-ruler {B_final:.4f} <= {G1.SHUT_BAR}) at "
                  f"healthy CE ({CE_final:.4f} <= root+0.30 "
                  f"{root_ce_r + G1.CE_NOISE_BAR:.4f}) while the reference "
                  f"(unwalled) leg installs B at "
                  f"{r_fin['B_gm12']:.4f} — the wall is a splint; rescope "
                  "the claim to 'the wall freezes the organism; memory "
                  "survives freezing'.")
    elif ZERO_SUM:
        verdict = "ZERO-SUM"
        clause = (f"B installs (B-ruler {B_final:.4f} >= {B_INSTALL_BAR}) and "
                  f"A dies (A-ruler {A_final:.4f} < {G1.MAINTAIN_BAR}) — "
                  "displacement inside the ball kills; the ball is a "
                  "single-exhibit museum by geometry.")
    elif B_final <= G1.SHUT_BAR and not ce_healthy:
        verdict = "AMBIGUOUS (B fails at UNhealthy CE)"
        clause = (f"B fails ({B_final:.4f} <= {G1.SHUT_BAR}) but CE_r "
                  f"{CE_final:.4f} exceeds root+0.30 "
                  f"({root_ce_r + G1.CE_NOISE_BAR:.4f}) — the MUSEUM bar's "
                  "'healthy CE' precondition does not hold; no bar fires.")
    else:
        verdict = "AMBIGUOUS (B in the (0.27, 0.7) gap)"
        clause = (f"B-ruler lands at {B_final:.4f}, strictly between the "
                  f"death bar {G1.SHUT_BAR} and the install bar "
                  f"{B_INSTALL_BAR} — partial install; no bar fires; the "
                  f"reference leg's B reads {r_fin['B_gm12']:.4f} for the "
                  "capacity comparison.")

    log("=" * 78)
    log(f"G1BW VERDICT: {verdict}")
    log(f"  walled final: A {A_final:.4f} | B {B_final:.4f} | CE_r "
        f"{CE_final:.4f} (root {root_ce_r:.4f}) | A_min {A_min:.4f}")
    log(f"  reference final: A {r_fin['A_gm12']:.4f} | B {r_fin['B_gm12']:.4f}"
        f" | CE_r {r_fin['ce_r']:.4f}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # FREE-RUN BATTERIES (both facts; root / walled / reference / rider)
    # =====================================================================
    log("free-run batteries (e043 R3 convention)")
    n_pr = 2 if SMOKE else 4
    prompts_a = ([train_text[p - 120: p] for p, _ in install_occ[:n_pr]]
                 + [train_text[p - 120: p] for p, _ in held_occ[:n_pr]])
    prompts_b = ([train_text[p - 120: p] for p, _ in install_occ_b[:n_pr]]
                 + [train_text[p - 120: p] for p, _ in held_occ_b[:n_pr]])
    fr_root = freerun_battery(G1.evl_load(theta0), corpus, prompts_a,
                              prompts_b, "root")
    fr_walled = freerun_battery(G1.evl_load(w_final_sd), corpus, prompts_a,
                                prompts_b, "walled_final")
    fr_ref = freerun_battery(G1.evl_load(r_final_sd), corpus, prompts_a,
                             prompts_b, "reference_final")

    # ---------------- THE ZERO-COMPUTE RIDER (existing checkpoint) ---------
    log(f"rider: free-run expression battery on {RIDER_CK.name} "
        f"(washed +300, NO second fact)")
    rider_raw = torch.load(RIDER_CK, map_location="cpu", weights_only=False)
    rider_sd = rider_raw["model"] if "model" in rider_raw else rider_raw
    rider_meta_in = rider_raw.get("meta", {}) if isinstance(rider_raw, dict) else {}
    rider_net = G1.evl_load(rider_sd)
    fr_rider = freerun_battery(rider_net, corpus, prompts_a, prompts_b,
                               "rider_W1_10907_s300")
    rider_cells = eval_dial(rider_sd, "rider")

    # ---------------- clean-judge scores (e110 machinery, LAZY import) -----
    judges = {"root": G1.load_g1(CKPT_DIR / GB.ROOT_CK),
              "reference_final_B": G1.evl_load(r_final_sd)}
    stream_pairs = []
    for fact, prompts in (("A", prompts_a), ("B", prompts_b)):
        for i, pr in enumerate(prompts):
            stream_pairs.append((f"{fact}{i}", pr,
                                 fr_walled[fact]["completions"][i]["continuation"]))
            stream_pairs.append((f"rider_{fact}{i}", pr,
                                 fr_rider[fact]["completions"][i]["continuation"]))
    judge = judge_battery(stream_pairs, judges)
    log(f"clean-judge (e110 machinery importable: {judge['importable']}): "
        + ("" if not judge["importable"] else
           " | ".join(f"{k} mean {v['mean']:.3f}" for k, v in judge.items()
                      if isinstance(v, dict) and "mean" in v)))

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    def cells_table(cells: dict) -> dict:
        return {str(s): c for s, c in sorted(cells.items())}

    metrics = {
        "experiment": "g1bW_second_fact",
        "date": common.now_iso(),
        "design": ("R56 critic's KILLER CONTROL (scratch/r56_critic.md lines "
                   "54-64, VERBATIM bars): the WALL's MUSEUM TEST — can a "
                   "walled net install a SECOND fact? W1 machinery on the "
                   "seed-10907 lineage; commit + 50 wash + B install (300 "
                   "steps, e043-Dmix) UNDER the projection (anchor stays at "
                   "A's commit); reference leg = the same install on the "
                   "unwalled washed control; zero-compute rider = free-run "
                   "battery on the existing g1bR_W1_10907_s300.pt"),
        "bars": G1BW_BARS,
        "question": ("does commit(0.7) + hard L2 projection — which holds "
                     "fact A's ruler flat at ~0.9 through the wash that kills "
                     "controls — admit a SECOND fact installed afterward "
                     "inside the same ball? SPLINT-REFUTED vs MUSEUM vs "
                     "ZERO-SUM."),
        "provenance": {
            "root": {"checkpoint": f"runs/checkpoints/{GB.ROOT_CK}",
                     "params": GB.G1B_PARAMS, "bitexact": G_BITEXACT},
            "wash": {"seed": WASH_SEED, "steps": int(WASH_STEPS_CKS[-1]),
                     "stream": "e170 neutral bank (seed 170) via g1_wash",
                     "lineage": "g1bR's first replicate seed (W1_10907: "
                                "+50 g-12 0.9157, +300 0.9056)"},
            "install": {"steps": int(INSTALL_STEPS),
                        "gen_seed": B_INSTALL_SEED,
                        "splice_rng": B_SPLICE_RNG,
                        "stream": "e043 Dmix VERBATIM (g1 phase-0a "
                                  "operationalization): 16 spliced + 16 "
                                  "paired originals + 32 random/step, "
                                  "full-token union CE, house cosine "
                                  "warmup 100, AdamW (0.9,0.95) wd 0.1 "
                                  "clip 1.0"},
            "fact_B": {"name": B_NAME, "hosts": B_HOSTS, "host_mix": b_mix,
                       "train_occ": b_train_occ, "val_occ": b_val_occ,
                       "onset_char": B_NAME[0],
                       "selection_rule": "first of [QUORINA, MERIDIA, "
                                         "KORVETH, XYRANNE] with 0 train+val "
                                         "occurrences"},
            "devices": {"walled_wash": w_wash["device"],
                        "walled_install": w_ins["device"],
                        "ref_wash": r_wash["device"],
                        "ref_install": r_ins["device"]},
            "runtimes_s": {"walled_wash": None, "walled_install":
                           w_ins["runtime_s"], "ref_install": r_ins["runtime_s"]},
        },
        "reference_leg": {
            "role": "capacity reference for MUSEUM vs ZERO-SUM (co-reported, "
                    "NOT gating)",
            "wash_gm12": r_wash_gm12,
            "install_cells": cells_table(r_ins["cells"]),
            "final": {"A_gm12": r_fin["A_gm12"], "B_gm12": r_fin["B_gm12"],
                      "B_g0": r_fin["B_g0"],
                      "B_held30_gm12": r_fin["B_held30_gm12"],
                      "ce_r": r_fin["ce_r"]},
            "freerun": fr_ref,
        },
        "walled_leg": {
            "wash_gm12": w_wash_gm12,
            "wash_traj": [{k: v for k, v in t.items()} for t in w_wash["traj"]],
            "install_cells": cells_table(w_ins["cells"]),
            "install_traj": w_ins["traj"],
            "final": {"A_gm12": A_final, "A_g0": w_fin["A_g0"],
                      "A_gp12": w_fin["A_gp12"],
                      "A_held30_gm12": w_fin["A_held30_gm12"],
                      "B_gm12": B_final, "B_g0": w_fin["B_g0"],
                      "B_gp12": w_fin["B_gp12"],
                      "B_held30_gm12": w_fin["B_held30_gm12"],
                      "ce_r": CE_final, "A_min_over_ckpts": A_min},
        },
        "root_cells": root_cells,
        "rider": {"checkpoint": str(RIDER_CK.relative_to(E43.REPO)).replace("\\", "/"),
                  "stored_meta": rider_meta_in,
                  "cells": rider_cells,
                  "freerun": fr_rider,
                  "note": "zero-compute: the existing +300-wash state (no "
                          "second fact) — the free-run expression reference "
                          "for A under the wall WITHOUT B's install"},
        "freerun": {"root": fr_root, "walled_final": fr_walled,
                    "reference_final": fr_ref, "rider": fr_rider,
                    "judge": judge},
        "gates": gates,
        "adjudication": {
            "bars_verbatim": G1BW_BARS,
            "gates_pass": gates_pass,
            "reads": {"A_final": A_final, "B_final": B_final,
                      "CE_final": CE_final, "root_ce_r": root_ce_r,
                      "ce_health_bar": root_ce_r + G1.CE_NOISE_BAR,
                      "B_installs": B_installs, "A_holds": A_holds,
                      "ce_healthy": ce_healthy, "A_min": A_min,
                      "read_at": f"install s{last_ck} (final state)"},
            "SPLINT_REFUTED": SPLINT_REFUTED,
            "MUSEUM": MUSEUM,
            "ZERO_SUM": ZERO_SUM,
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "n1_lineage": ("one root, one lineage (e131 line), ONE wash seed "
                           "(10907, g1bR's first replicate): the wall's "
                           "wash-robustness is licensed at n=3 by g1bR, but "
                           "THIS cell's B-fold is n=1 per leg."),
            "b_draw_n1": (f"B is ONE draw: one name ({B_NAME}), one host "
                          f"pool ({b_mix}), one install seed "
                          f"({B_INSTALL_SEED}), one splice RNG "
                          f"({B_SPLICE_RNG}) — B's installability may be "
                          "draw-sensitive (e043's own dose ladder showed "
                          "step-indexed ceilings); the reference leg shares "
                          "the IDENTICAL draw, so the walled-vs-reference "
                          "CONTRAST is draw-controlled even though B's "
                          "absolute level is n=1."),
            "device": ("the gates serialize against g2f; devices recorded per "
                       "training in provenance.devices; g1bR's own arms "
                       "migrated CPU mid-run — cross-device float fuzz is "
                       "covered by G-WASHREP's 0.05 tolerance and the bars' "
                       "margins."),
            "bars_read_at_end": ("all three bars read the FINAL install "
                                 "state; the per-checkpoint tables are "
                                 "co-reported so any transient fold (e.g. B "
                                 "installs then decays, or A dips mid-install "
                                 "and recovers) is visible, not hidden."),
            "no_bar_shopping": ("the thresholds (0.7 / 0.5 / 0.27 / "
                                "root+0.30) were registered in this file's "
                                "docstring before the run; the (0.27, 0.7) "
                                "gap and the unhealthy-CE case are reported "
                                "as AMBIGUOUS, never re-tuned."),
            "rider_is_not_a_control": ("the rider checkpoint was produced by "
                                       "g1bR (its own seeds/devices); it is "
                                       "an expression reference, not a "
                                       "same-process control."),
        },
        "trims": G1.trims, "deviations": deviations,
        "device_events": G1.device_events,
        "device_policy": {"parked": G1.GPU_PARKED, "reason": G1.PARK_REASON,
                          "wait_gate_s": GPU_WAIT_MAX,
                          "cooldown_s": COOLDOWN_S},
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": GB.G1B_PARAMS,
                   "R": R_CLAIM, "wash_seed": WASH_SEED,
                   "install_steps": int(INSTALL_STEPS),
                   "B": {"name": B_NAME, "hosts": B_HOSTS,
                         "splice_rng": B_SPLICE_RNG,
                         "gen_seed": B_INSTALL_SEED},
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "museum_test.png", w_ins, r_ins, w_wash_gm12, r_wash_gm12,
         root_cells, rider_cells, verdict, clause, gates_pass, A_final,
         B_final, R_CLAIM)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'museum_test.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1bW_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, w_ins, r_ins, w_wash_gm12, r_wash_gm12, root_cells,
         rider_cells, verdict, clause, gates_pass, A_final, B_final, R):
    """THE MUSEUM TEST figure: A and B rulers vs step (walled vs reference),
    CE_r, the critic's collapse channels, displacement vs the wall, verdict."""
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    w_keys = sorted(w_ins["cells"])
    r_keys = sorted(r_ins["cells"])
    w_x = [int(s) for s in w_keys]
    r_x = [int(s) for s in r_keys]

    # (0,0) A-RULER: walled vs reference across wash + install
    ax = axes[0, 0]
    wsh_x = sorted(w_wash_gm12)
    ax.plot(wsh_x, [w_wash_gm12[s] for s in wsh_x], "s-", ms=7, lw=2.2,
            color="seagreen", alpha=0.95, label="WALLED — wash phase (A ruler)")
    ax.plot(w_x, [w_ins["cells"][s]["A_gm12"] for s in w_keys], "s-", ms=7,
            lw=2.4, color="seagreen", label="WALLED — B install phase")
    rsh_x = sorted(r_wash_gm12)
    ax.plot(rsh_x, [r_wash_gm12[s] for s in rsh_x], "o--", ms=5, lw=1.4,
            color="crimson", alpha=0.6, label="REFERENCE — wash (no wall)")
    ax.plot(r_x, [r_ins["cells"][s]["A_gm12"] for s in r_keys], "o--", ms=5,
            lw=1.4, color="crimson", alpha=0.6,
            label="REFERENCE — B install (no wall)")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8)
    ax.axhline(root_cells["A_gm12"], ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.annotate(f"root {root_cells['A_gm12']:.3f}", (0, root_cells["A_gm12"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("wash steps | install steps (phase boundary at the gap)")
    ax.set_ylabel("A-ruler g-12 (mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("FACT A's RULER through the second install — walled vs "
                 "reference", fontsize=10)

    # (0,1) B-RULER: the museum test's own dial
    ax = axes[0, 1]
    ax.plot(w_x, [w_ins["cells"][s]["B_gm12"] for s in w_keys], "D-", ms=8,
            lw=2.4, color="royalblue", label="WALLED (R=0.7) — B install")
    ax.plot(r_x, [r_ins["cells"][s]["B_gm12"] for s in r_keys], "o--", ms=6,
            lw=1.8, color="darkorange", label="REFERENCE (no wall) — B install")
    ax.axhline(B_INSTALL_BAR, ls="--", lw=1.4, color="seagreen", alpha=0.9,
               label=f"B installs bar {B_INSTALL_BAR}")
    ax.axhline(G1.SHUT_BAR, ls="--", lw=1.1, color="tab:purple", alpha=0.8,
               label=f"B fails bar {G1.SHUT_BAR}")
    ax.axhline(root_cells["B_gm12"], ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.annotate(f"base {root_cells['B_gm12']:.4f}", (0, root_cells["B_gm12"]),
                textcoords="offset points", xytext=(6, -10), fontsize=7.5)
    ax.annotate(f"walled final {B_final:.3f}", (0.02, B_final),
                xycoords=("axes fraction", "data"), fontsize=8.5,
                weight="bold", color="royalblue")
    ax.set_xlabel("B install step")
    ax.set_ylabel(f"B-ruler g-12 (mean p({B_NAME[0]}), B install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title(f"THE MUSEUM TEST — can fact B ({B_NAME}) install inside "
                 "the ball?", fontsize=10)

    # (0,2) CE_r
    ax = axes[0, 2]
    ax.plot(w_x, [w_ins["cells"][s]["ce_r"] for s in w_keys], "s-", ms=6,
            lw=2.0, color="seagreen", label="WALLED")
    ax.plot(r_x, [r_ins["cells"][s]["ce_r"] for s in r_keys], "o--", ms=5,
            lw=1.6, color="darkorange", label="REFERENCE")
    ax.axhline(root_cells["ce_r"], ls=":", lw=1.2, color="gray",
               label=f"root CE_r {root_cells['ce_r']:.3f}")
    ax.axhline(root_cells["ce_r"] + G1.CE_NOISE_BAR, ls="--", lw=1.1,
               color="seagreen", alpha=0.8,
               label=f"health bar root+0.30 "
                     f"{root_cells['ce_r'] + G1.CE_NOISE_BAR:.3f}")
    ax.set_xlabel("B install step")
    ax.set_ylabel("CE_r (root-stream val bank)")
    ax.legend(fontsize=7.5)
    ax.set_title("ROOT-STREAM CE — the 'healthy CE' precondition", fontsize=10)

    # (1,0) THE CRITIC'S COLLAPSE CHANNELS (channel rider)
    ax = axes[1, 0]
    ax.plot(w_x, [w_ins["cells"][s]["row0_strength"] for s in w_keys], "s-",
            ms=6, lw=2.0, color="seagreen", label="WALLED row0 strength")
    ax.plot(w_x, [w_ins["cells"][s]["band121_129_max"] for s in w_keys], "^-",
            ms=6, lw=1.8, color="royalblue",
            label="WALLED band121-129 max (A's wpe band)")
    ax.plot(r_x, [r_ins["cells"][s]["row0_strength"] for s in r_keys], "o--",
            ms=4, lw=1.2, color="crimson", alpha=0.5,
            label="REFERENCE row0")
    ax.axhline(root_cells["row0_strength"], ls=":", lw=1.0, color="gray",
               label=f"root row0 {root_cells['row0_strength']:+.3f}")
    ax.axhline(root_cells["band121_129_max"], ls=":", lw=1.0, color="navy",
               alpha=0.6, label=f"root band max "
                                f"{root_cells['band121_129_max']:+.3f}")
    ax.set_xlabel("B install step")
    ax.set_ylabel("census strength (min-arm/zero-arm, A g0 readout)")
    ax.legend(fontsize=7)
    ax.set_title("CHANNEL RIDER — what the wall protected during B's install",
                 fontsize=10)

    # (1,1) DISPLACEMENT vs the wall
    ax = axes[1, 1]
    w_disp = [(i + 1, t["cum_disp"]) for i, t in enumerate(w_ins["traj"])
              if (i + 1) in w_ins["cells"]]
    ax.plot([p[0] for p in w_disp], [p[1] for p in w_disp], "s-", ms=6,
            lw=2.0, color="seagreen", label="WALLED raw |d| vs theta0")
    r_disp = [(i + 1, t["cum_disp"]) for i, t in enumerate(r_ins["traj"])
              if (i + 1) in r_ins["cells"]]
    ax.plot([p[0] for p in r_disp], [p[1] for p in r_disp], "o--", ms=5,
            lw=1.6, color="crimson", alpha=0.7,
            label="REFERENCE raw |d| (free)")
    ax.axhline(R, ls="--", lw=1.2, color="k", alpha=0.7, label=f"R={R}")
    ax.axhline(R + G1.PIN_FUZZ_BAR, ls=":", lw=1.0, color="k", alpha=0.5,
               label=f"G-PIN bound {R + G1.PIN_FUZZ_BAR}")
    ax.set_xlabel("B install step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_{anchor}\|_2$ at checkpoints")
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("DISPLACEMENT vs THE BALL (anchor = A's commit)", fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1BW — THE WALL'S MUSEUM TEST (seed-10907 lineage)",
            fontsize=10.5, va="top", family="monospace", weight="bold")
    y -= 0.052
    ax.text(0.02, y, f"gates: {'ALL PASS' if gates_pass else 'FAILURE'} | "
            f"commit({R}) + 50 wash (seed 10907) + {B_NAME} install 300 "
            f"UNDER the projection", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    ax.text(0.02, y, f"  WALLED   final: A {A_final:.4f} | B {B_final:.4f} "
            f"(bars 0.5 / 0.7 / 0.27)", fontsize=7.5, va="top",
            family="monospace", weight="bold", color="royalblue")
    y -= 0.032
    ax.text(0.02, y, f"  REFERENCE final: B {r_ins['cells'][sorted(r_ins['cells'])[-1]]['B_gm12']:.4f}"
            f" | A {r_ins['cells'][sorted(r_ins['cells'])[-1]]['A_gm12']:.4f}"
            f" (capacity reference)", fontsize=7.2, va="top",
            family="monospace", color="darkorange")
    y -= 0.032
    ax.text(0.02, y, f"  rider (+300 wash, no B): A {rider_cells['A_gm12']:.4f}"
            f" | freerun A-name x"
            f"{sum(c['counts']['name'] for c in [metrics_fr_rider_placeholder]) if False else ''}"
            f"{rider_fr_counts if False else ''}", fontsize=7.2, va="top",
            family="monospace", color="gray")
    y -= 0.042
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.042
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.026

    fig.suptitle("G1BW — THE WALL'S MUSEUM TEST: can a walled net install a "
                 f"SECOND fact ({B_NAME})? -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# placeholder globals used only by the verdict panel's rider line
rider_fr_counts = None
metrics_fr_rider_placeholder = None


if __name__ == "__main__":
    sys.exit(main())
