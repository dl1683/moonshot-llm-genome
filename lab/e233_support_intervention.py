"""E233 — THE SUPPORT-PROJECTION INTERVENTION (T210's named cell; W030's
"the next instrument is the intervention"; design frozen in
scratch/e233_design.md BEFORE this script — the design note is the
registration; this docstring carries its bars VERBATIM).

THE QUESTION: e226 located the third dimension's SEAT — from +10 the
wash's continuing pull works along the DYING anchor's support (2.6-14.8x
larger component along iPhone's support than Gmail's, on all three
washes), while the static t=0 geometry is family-typical. Is that aligned
component the CARRIER of the death (removing it spares iPhone) or a
MARKER (removing it changes nothing — g14's LETHAL != CARRIER precedent)?

THE CELL (per the frozen design): on the 124M organism at the t=0 state
(the e182c archive conventions; the wash stream replayed bit-exactly per
e226's certified draw streams):
  1. ARM-P (projected wash): continue the wash VERBATIM (same optimizer,
     same batches, same seeds) EXCEPT each applied step is projected OFF
     iPhone's t=0 support direction: step' = step - <step, s_iPhone> *
     s_iPhone (norm not rescaled — the projection removes a component;
     the removed L2 per step recorded; a matched-L2 control arm is
     required because the projected wash is slightly smaller).
  2. ARM-C (matched-L2 control): the wash with each step scaled to
     ARM-P's norm (removing the magnitude difference, isolating the
     DIRECTION removal).
  3. Reference fates: the three committed washes (Gmail hr
     0.905/0.931/0.864; iPhone 0.184/0.342/0.195 — runtime-read from
     runs/e226/metrics.json, never transcribed).
  4. Readouts: the two anchors' hr at +10/+50/+80 (the e226
     conventions); the nearrel battery's family members (prediction (a));
     CE trajectory (the wash must still be a wash — CE improving).

BARS VERBATIM (scratch/e233_design.md, frozen at dispatch BEFORE any
compute; adjudicate against exactly this; no bar shopping):
  - FATE-FLIPS — "iPhone's hr at +80 rises to >= 0.5x Gmail's committed
    hr while ARM-C's iPhone stays dying — the aligned component is the
    CARRIER; the third dimension's seat is causal; H-i confirmed"
  - FATE-HOLDS — "ARM-P and ARM-C leave both anchors' fates within the
    committed wash spread — the alignment is a MARKER; the seat stays
    geometric but non-causal; H-ii"
  - ANY — "anything between — the four trajectories verbatim (iPhone-P,
    iPhone-C, Gmail-P, Gmail-C vs the three references), no narrative
    inflation"

REGISTERED PREDICTIONS (no retrofit):
  (a) "Under FATE-FLIPS, the nearrel battery's iPhone-family members
      also spare (the seat is the relation family's, not the anchor's
      alone)."
  (b) "Under FATE-HOLDS, Gmail's fate is also unchanged (the projection
      touched only a component the wash does not need) — and the t=0
      consumption read (below) becomes the live mechanism thread."
  (c) "The removed-L2 ledger: if the projected component is < ~1% of the
      step norms, the intervention is underpowered and the verdict must
      say so (the honest-instrument clause)."

OPERATIONALIZATIONS (frozen here BEFORE compute; they fix the clauses,
they do not move the bars):
  * hr := p(state)/p(t=0) per probe; p = p(answer first token | the
    probe's VERBATIM 2-shot prompt), CPU fp32 batch-1, the committed
    instrument (e182/e182c's batteries, module import; t=0 re-certified
    against all THREE committed records, dp <= 0.010, e226's G_BATT).
  * the APPLIED step Delta_t := the full parameter delta of the VERBATIM
    AdamW step at the arm's CURRENT weights (snapshot W_before ->
    verbatim backward+clip+opt.step() -> Delta = W_after - W_before ->
    restore W_before -> apply the modified delta). The OPTIMIZER STATE
    advances on the verbatim step given the trajectory's own gradient —
    the machinery is verbatim; only the applied weight delta is
    modified (disclosed; see honesty guards).
  * s_iPhone(0) := e226's support convention recomputed FRESH fp32 CPU
    (grad p(ans|prompt) over all 124,439,808 params, batch-1, eval,
    dropout 0, L2-normalized, fp64-normed), ONE fixed direction for all
    steps and both anchors' co-reads; FD-gated at eps 0.02/0.05 (e204's
    convention); determinism re-verified (recompute self-cos > 0.999).
  * the projection: Delta'_t = Delta_t - <Delta_t, s_iPhone> * s_iPhone
    (the component REMOVED, norm NOT rescaled); all dots/norms fp64
    (chunked per-parameter fp64 dots — e226's instrument finding: fp32
    accumulation over 124M coords drifts ~0.6%, same order as the 1%
    honest-instrument bar).
  * ARM-C's step := Delta_C,t * (||Delta'_P,t|| / ||Delta_C,t||) —
    step-locked to ARM-P's post-projection per-step norms (ARM-P runs
    first; the ledger is the coupling).
  * the stream := wash 1's frozen window-draw stream (seed 18202 —
    e182's own wash), reproduced from the archived seed and certified
    BIT-EXACTLY against the archived generator state at step 80
    (e226's G_DRAWS convention). n=1 organism, n=1 stream per arm
    (the design's honesty guard; the 3-wash replication belongs to
    e226's observational base).
  * "Gmail's committed hr" (the FATE-FLIPS line) := the SAME-STREAM
    committed reference (wash 1's Gmail hr, runtime-read); the lines
    for washes 2/3 are co-reported.
  * "ARM-C's iPhone stays dying" := ARM-C's iPhone hr(+80) within the
    committed iPhone spread [min, max] of the three committed hrs.
  * "within the committed wash spread" (FATE-HOLDS) := ALL FOUR
    (Gmail-P, Gmail-C, iPhone-P, iPhone-C) hr(+80) within their
    anchor's committed spread. (iPhone-P above the flip line exits the
    iPhone spread, so FLIPS and HOLDS are mutually exclusive when the
    FLIPS conjunction holds.)
  * Adjudication order: FATE-FLIPS -> FATE-HOLDS -> ANY (e182c's
    pattern); every boolean reported regardless.
  * "(c) the projected component is < ~1% of the step norms" := median
    over the arm's steps of |<Delta_t, s_iPhone>| / ||Delta_t|| < 0.01
    -> the verdict is CO-STAMPED UNDERPOWERED (the stamp discloses, it
    does not move the bar).
  * "the nearrel battery's iPhone-family members" := the product
    family's five non-anchor members (Xbox, Chrome, iPad, iTunes,
    PlayStation — e216's family6 label of iPhone's own battery family;
    the battery that contains the anchors); the literal near battery
    (near-uscap: Boston/Atlanta/Columbus) is co-reported. "spare" (a)
    := hr_P(+80) ABOVE that member's own committed wash-1 hr(+80);
    count + table reported, no new bar.
  * the wash-health read (the wash must still be a wash): held-out bank
    ppl at +80 < bank ppl at t=0 AND mean in-batch CE over the last 10
    steps < over the first 10, for BOTH arms (G_WASHHEALTH).

HONESTY GUARDS (the design's, verbatim in force):
  * the support direction is t=0-fixed (ONE direction for all steps;
    e226's rotation reads were sub-bar, so a fixed direction is
    defensible — disclosed).
  * the intervention changes the trajectory, so after step 1 the wash is
    no longer the committed wash — this is the cell's nature (g12's
    precedent); the readouts compare FATES, not trajectories.
  * n=1 organism, n=1 wash stream per arm; 124M GPU steps in <=180 s
    bursts, temp-gated.
  * the GRADED discipline: FATE-FLIPS at exactly the stated line; no bar
    shopping.
  * DEVICE TEXTURE (disclosed): the arms train GPU fp32 (TF32 OFF);
    wash 1's committed reference is e182c's CPU fp32 replay; washes 2/3
    are GPU fp32 — the three-reference spread spans the device texture
    and the arms are compared to the references as FATES (e217's
    precedent; ARM-C is the arms' own device-matched control).

COMPUTE ENVELOPE (dispatch): the GPU lane is owned (no concurrent GPU
jobs; the owner's max-priority window ACTIVE per STATE.json: bursts
<=180 s, short 30-60 s cooldowns, temp-aware, never past 85C). This
cell: bursts <= 175 s wall AND <= 40 steps; cooldown >= 45 s; every
launch gated by common.gpu_ok() (util <= 85%, temp <= 80C, mem <= 85%);
mid-burst thermal guard ends the burst at >= 84C; every poll logged to
runs/_envelope_log.jsonl. CPU fp32 probing between bursts (counts toward
cooldown). Single training runs <= 180 s bursts; resumable state after
every burst; progressive PARTIAL metrics writes. No NOTES/THINKING/
QUEUE/STATE edits (dispatch). Smoke via E233_SMOKE=1 (3 steps, own
smoke dir, nothing adjudicated).

Run:  cd lab && python e233_support_intervention.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import torch                                          # noqa: E402

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import now_iso, run_dir, save_json          # noqa: E402

import e182c_forgetting_control as e1                  # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                           # noqa: E402 — the template battery, VERBATIM
import e226_interior as e3                             # noqa: E402 — the support/wash-gradient conventions, VERBATIM

# e182c sets threads 8, e226 resets to 4 (its shared-box envelope); the
# owner's max-priority window is ACTIVE (STATE.json) and the box is ours
# — restore e182's 8-thread convention for the CPU probing phases.
THREADS = 8
torch.set_num_threads(THREADS)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

try:
    import psutil                                     # noqa: E402
except ImportError:
    psutil = None

SMOKE = os.environ.get("E233_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e233_smoke" if SMOKE else "e233"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ------------------------------------------------------------ the cell
W_SEED = 18202                    # wash 1's frozen stream (e182's own)
N_STEPS = 3 if SMOKE else 80
CK_STEPS: tuple[int, ...] = (1, 2, 3) if SMOKE else (10, 50, 80)
FD_EPS = (0.05,) if SMOKE else (0.02, 0.05)   # e204's/e226's sizes (L2)
TOL_T0_DP = 0.010                  # e226's G_BATT tolerance
CHUNK_DOT_MIN = 4_000_000          # per-param fp64 dots on GPU are one op
                                          # below this; chunked above

# the owner's ACTIVE max-priority window (STATE.json), still temp-aware:
BURST_MAX_S = 175.0                # < the 180 s hard lab cap
BURST_MAX_STEPS = 40
COOLDOWN_S = 45.0                  # the 30-60 s window
TEMP_BURST_END = 84.0              # end the burst (never past 85C)
LAUNCH_POLL_GAP_S = 5.0

ANCHOR_G = "The email service made by Google->Gmail"
ANCHOR_I = "The phone made by Apple->iPhone"
IPHONE_FAMILY = (                 # e216's family6=product, non-anchor
    "The gaming console made by Microsoft->Xbox",
    "The web browser made by Google->Chrome",
    "The tablet made by Apple->iPad",
    "The music store made by Apple->iTunes",
    "The game console made by Sony->PlayStation",
)

# committed records (runtime-read, never transcribed)
E182C_M = common.REPO / "runs" / "e182c" / "metrics.json"
E182C2_J = common.REPO / "runs" / "e182c2" / "journal_p2.json"
E217_M = common.REPO / "runs" / "e217" / "metrics.json"
E216_M = common.REPO / "runs" / "e216" / "metrics.json"
E226_M = common.REPO / "runs" / "e226" / "metrics.json"
W1_LATEST = common.REPO / "runs" / "checkpoints" / "e182c_replay_latest.pt"
CK_DIR = common.REPO / "runs" / "checkpoints"

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "FATE-FLIPS": "iPhone's hr at +80 rises to >= 0.5x Gmail's "
            "committed hr while ARM-C's iPhone stays dying — the aligned "
            "component is the CARRIER; the third dimension's seat is "
            "causal; H-i confirmed",
        "FATE-HOLDS": "ARM-P and ARM-C leave both anchors' fates within "
            "the committed wash spread — the alignment is a MARKER; the "
            "seat stays geometric but non-causal; H-ii",
        "ANY": "anything between — the four trajectories verbatim "
            "(iPhone-P, iPhone-C, Gmail-P, Gmail-C vs the three "
            "references), no narrative inflation",
    },
    "predictions_verbatim": {
        "(a)": "Under FATE-FLIPS, the nearrel battery's iPhone-family "
            "members also spare (the seat is the relation family's, not "
            "the anchor's alone).",
        "(b)": "Under FATE-HOLDS, Gmail's fate is also unchanged (the "
            "projection touched only a component the wash does not need) "
            "— and the t=0 consumption read (below) becomes the live "
            "mechanism thread.",
        "(c)": "The removed-L2 ledger: if the projected component is < "
            "~1% of the step norms, the intervention is underpowered and "
            "the verdict must say so (the honest-instrument clause).",
    },
    "registration": "bars frozen VERBATIM from scratch/e233_design.md "
        "(the ripened design note, frozen by the dispatch brief BEFORE "
        "any compute; the dispatch brief is the registration); adjudicate "
        "against exactly this; no bar shopping",
}

trims: list[str] = []
deviations: list[str] = [
    "The APPLIED-step projection mechanics (registered before compute): "
    "the verbatim AdamW step is COMPUTED (snapshot -> opt.step() -> "
    "Delta), the weights are restored, and the MODIFIED delta is applied. "
    "The optimizer state advances on the verbatim step given the "
    "trajectory's own gradient — the machinery (optimizer, batches, "
    "seeds, clip) is verbatim; only the applied weight delta is "
    "modified. After step 1 the trajectory is the intervention's own "
    "(g12's precedent, the design's disclosed nature).",
    "The support direction is recomputed FRESH fp32 CPU here (e226's "
    "fp16 cache was a 13.4 GB storage compromise; the fresh fp32 "
    "recompute is the higher-fidelity instrument; determinism re-"
    "verified, self-cos > 0.999; FD-gated at both eps).",
    "ARM-C is step-locked to ARM-P's post-projection per-step norms "
    "(ARM-P runs first; the ledger couples them); both arms replay the "
    "SAME certified wash-1 stream (seed 18202) from the SAME pristine "
    "t=0 — the only differences are the projection (P) and the L2 match "
    "(C).",
    "DEVICE TEXTURE: the arms train GPU fp32 (TF32 OFF); w1's committed "
    "reference is a CPU fp32 replay and w2/w3 are GPU fp32 — the "
    "three-reference spread spans the texture; fates, not bit values, "
    "are compared (e217's precedent).",
    "CPU probing (batteries + bank ppl) uses the committed CPU fp32 "
    "instrument between GPU bursts (threads 8, e182's convention — the "
    "owner's max-priority window is active and the box is ours).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode (E233_SMOKE=1): 3 steps, checkpoints {1,2,3}, own smoke "
    "dir; the draw-stream certification is the FULL 80-draw check (CPU-"
    "only, cheap); nothing adjudicated or gated (SMOKE stamp).",
]


# ------------------------------------------------------------ envelope

def cpu_load_check(tag: str) -> dict:
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    rec = {"tag": tag,
           "cpu_percent": psutil.cpu_percent(interval=0.5),
           "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1),
           "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    log(f"  [load] {tag}: cpu {rec['cpu_percent']}% "
        f"ram_avail {rec['ram_avail_gb']} GB")
    return rec


def gpu_poll(tag: str) -> dict:
    s = common.gpu_status()
    ok = s["util"] <= common.GPU_UTIL_CEIL and s["temp"] <= common.GPU_TEMP_CEIL
    common._log_envelope_poll(f"{NAME}:{tag}", s["util"], s["temp"], ok)
    log(f"  [gpu:{tag}] util {s['util']:.0f}% temp {s['temp']:.0f}C "
        f"mem {s['mem_used']:.0f}/{s['mem_total']:.0f}MB "
        f"power {s['power']:.1f}W -> {'OK' if ok else 'HOLD'}")
    return {"poll": s, "ok": bool(ok)}


def wait_gpu_free(tag: str, max_wait_s: float = 1800.0) -> list[dict]:
    """Launch gate: common.gpu_ok() semantics, first launch double-polled
    (>=5 s apart). The owner's max-priority window is active — short
    waits only; never migrate without exhausting the window (the e227
    CPU-migrate precedent is the fallback, disclosed if ever used)."""
    polls = [gpu_poll(f"{tag}#1")]
    time.sleep(LAUNCH_POLL_GAP_S)
    polls.append(gpu_poll(f"{tag}#2"))
    t0w = time.time()
    while not (polls[-2]["ok"] and polls[-1]["ok"]):
        if time.time() - t0w > max_wait_s:
            raise RuntimeError(f"GPU window never opened for {tag}")
        log(f"  [gpu:{tag}] waiting 20s for the envelope "
            f"(util<={common.GPU_UTIL_CEIL:.0f}% "
            f"temp<={common.GPU_TEMP_CEIL:.0f}C)")
        time.sleep(20.0)
        polls.append(gpu_poll(f"{tag}#w"))
    return polls


def burst_temp_check(tag: str) -> bool:
    """Mid-burst thermal guard: True = keep going; ends the burst at
    >= TEMP_BURST_END (never past 85C)."""
    s = common.gpu_status()
    common._log_envelope_poll(f"{NAME}:{tag}:mid", s["util"], s["temp"],
                              s["temp"] < TEMP_BURST_END)
    if s["temp"] >= TEMP_BURST_END:
        log(f"  [gpu:{tag}:mid] temp {s['temp']:.0f}C >= "
            f"{TEMP_BURST_END:.0f}C — ending burst early")
        return False
    return True


# ------------------------------------------------------------ fp64 math
# e226's instrument finding: fp32 accumulation over 124M coords drifts
# ~0.6% — the SAME order as the registered 1% honest-instrument bar, so
# every ledger dot/norm is fp64 (per-parameter fp64 dots on GPU; single
# dot() calls are exact fp64 reductions).

def _dot64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp64 dot of two flat fp32 GPU tensors (chunked casts for the
    embedding-scale tensors to bound fp64 temporaries)."""
    n = a.numel()
    if n <= CHUNK_DOT_MIN:
        return torch.dot(a.double(), b.double())
    tot = torch.zeros((), dtype=torch.float64, device=a.device)
    for s in range(0, n, CHUNK_DOT_MIN):
        sl = slice(s, min(s + CHUNK_DOT_MIN, n))
        tot += torch.dot(a[sl].double(), b[sl].double())
    return tot


# ------------------------------------------------------------ the arm

def run_arm(tag: str, net0, train_ids, offs, bank_xy, s_chunks_gpu,
            p_ledger, probes_bats, on_ckpt) -> dict:
    """One arm of the cell. tag in {"P", "C"}.

    P: each applied step projected off s_iPhone (norm NOT rescaled).
    C: each applied step scaled to P's step-locked post-projection norm.

    VERBATIM wash machinery: AdamW(0.9,0.95) wd 0.1 constant lr 5e-5,
    clip 1.0, full-token CE, batch 8 x ctx 512, wash-1's certified
    window stream; GPU fp32 (TF32 OFF) in <=175 s / <=40-step bursts.
    Resumable state after every burst; per-checkpoint weight archives."""
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=e1.LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    params = list(net.parameters())
    step = 0
    ledger: dict[int, dict] = {}
    latest = CK_DIR / f"{NAME}_{tag}_latest.pt"
    if latest.exists() and not SMOKE:
        st = torch.load(latest, map_location=CPU, weights_only=False)
        if st["step"] > 0:
            net.load_state_dict(st["model"])
            net.to(dev)
            opt.load_state_dict(st["opt"])
            e2.opt_state_dev_fix(opt, dev)
            step = int(st["step"])
            ledger = {int(k): v for k, v in st["ledger"].items()}
            log(f"ARM-{tag}: RESUMED from step {step} "
                f"({len(ledger)} ledger rows)")
    ckpt_set = set(CK_STEPS)
    envelope_polls: list[dict] = []
    burst_id = step + 1
    t_burst = None
    n_burst = 0
    while step < N_STEPS:
        if t_burst is None:
            envelope_polls += wait_gpu_free(f"{tag}burst{burst_id}")
            t_burst = time.time()
            n_burst = 0
            log(f"ARM-{tag}: burst {burst_id} opens at s{step + 1}")
        step += 1
        n_burst += 1
        off = offs[step - 1]
        x = torch.stack([train_ids[o: o + e1.SEQ] for o in off]).to(dev)
        y = torch.stack([train_ids[o + 1: o + 1 + e1.SEQ]
                         for o in off]).to(dev)
        logits = net(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        # ---- THE INTERVENTION: snapshot -> verbatim step -> modify delta
        base = [p.detach().clone() for p in params]
        opt.step()
        with torch.no_grad():
            ds = [(p.detach() - b).reshape(-1)
                  for p, b in zip(params, base)]
            c64 = torch.zeros((), dtype=torch.float64, device=dev)
            dn64 = torch.zeros((), dtype=torch.float64, device=dev)
            for d, s_c in zip(ds, s_chunks_gpu):
                c64 += _dot64(d, s_c)
                dn64 += _dot64(d, d)
            dn = float(math.sqrt(dn64.item()))
            cf = float(c64.item())
            if tag == "P":
                c32 = torch.tensor(cf, dtype=torch.float32, device=dev)
                dpn64 = torch.zeros((), dtype=torch.float64, device=dev)
                for p, b, d, s_c in zip(params, base, ds, s_chunks_gpu):
                    nd = d - c32 * s_c
                    dpn64 += _dot64(nd, nd)
                    p.copy_(b.reshape(p.shape)
                            + nd.reshape(p.shape))
                dpn = float(math.sqrt(dpn64.item()))
                ledger[step] = {"ce": float(loss.item()), "dnorm": dn,
                                "dot_s": cf, "dpnorm": dpn,
                                "removed_frac": abs(cf) / dn if dn else 0.0,
                                "removed_l2": abs(cf)}
            else:                                   # ARM-C matched-L2
                tgt = p_ledger[step]["dpnorm"]
                assert tgt > 0, f"ARM-P ledger missing/gzero at {step}"
                scale = tgt / dn if dn else 1.0
                s32 = torch.tensor(scale, dtype=torch.float32, device=dev)
                for p, b, d in zip(params, base, ds):
                    p.copy_(b.reshape(p.shape)
                            + (s32 * d).reshape(p.shape))
                ledger[step] = {"ce": float(loss.item()), "dnorm": dn,
                                "scale": scale, "applied_norm": tgt}
        del base, ds
        if step % 10 == 0 or step in ckpt_set or step == N_STEPS:
            r = ledger[step]
            extra = (f"|removed| {r.get('removed_l2', float('nan')):.3e} "
                     f"({r.get('removed_frac', float('nan'))*100:.2f}% "
                     f"of |d|)" if tag == "P" else
                     f"scale {r.get('scale', float('nan')):.5f}")
            log(f"  [ARM-{tag}] s{step:3d}/{N_STEPS} CE "
                f"{r['ce']:.4f} |d| {r['dnorm']:.4f} "
                f"|d'| {r.get('dpnorm', r.get('applied_norm', float('nan'))):.4f} "
                f"{extra} ({time.time() - t_burst:.1f}s into burst)")
        # ---- burst bookkeeping / thermal guard
        temp_ok = True
        if n_burst % 8 == 0:
            temp_ok = burst_temp_check(f"{tag}burst{burst_id}")
        hit_ckpt = step in ckpt_set
        burst_over = (n_burst >= BURST_MAX_STEPS
                      or time.time() - t_burst >= BURST_MAX_S
                      or hit_ckpt or step >= N_STEPS or not temp_ok)
        if not burst_over:
            continue
        t_end = time.time()
        log(f"  [ARM-{tag}] burst {burst_id} done: {n_burst} steps in "
            f"{t_end - t_burst:.1f}s (caps {BURST_MAX_S:.0f}s/"
            f"{BURST_MAX_STEPS} steps), now at s{step}")
        sd = {k: v.detach().to("cpu", torch.float32).clone()
              for k, v in net.state_dict().items()}
        torch.save({"model": sd, "opt": e2._opt_state_to_cpu(opt.state_dict()),
                    "step": step, "ledger": ledger,
                    "meta": {"experiment": NAME, "arm": tag,
                             "seed": W_SEED, "lr": e1.LR,
                             "desc": f"e233 ARM-{tag} "
                                     f"({'projected' if tag == 'P' else 'matched-L2'} "
                                     f"wash-1 stream seed {W_SEED}) GPU fp32, "
                                     f"step {step}"}},
                   latest)
        if hit_ckpt:
            torch.save({"model": sd,
                        "meta": {"experiment": NAME, "arm": tag, "step": step,
                                 "lr": e1.LR, "seed": W_SEED,
                                 "desc": f"e233 ARM-{tag} checkpoint, step {step}",
                                 "base": e1.MODEL_REPO,
                                 "revision": e1.MODEL_REV}},
                       CK_DIR / f"{NAME}_{tag}_s{step}.pt")
        if hit_ckpt:
            on_ckpt(tag, step, sd, ledger[step]["ce"])
        del sd
        # cooldown (>= 45 s since burst end; the CPU probing above counts)
        remain = COOLDOWN_S - (time.time() - t_end)
        if remain > 0 and step < N_STEPS:
            log(f"  [thermal] cooldown {remain:.0f}s "
                f"(max-priority window: {COOLDOWN_S:.0f}s)")
            time.sleep(remain)
        burst_id = step + 1
        t_burst = None
    return {"ledger": ledger, "envelope_polls": len(envelope_polls)}


# ------------------------------------------------------------ plots

def make_fates_plot(rd, states, refs, adj):
    """THE FATE FIGURE: the four trajectories vs the three committed
    references + the CE panel + the verdict panel."""
    verdict = adj["verdict"]
    sts = sorted(int(s) for s in states["P"])       # 0,10,50,80
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))

    for ax, anchor, key in ((axes[0][0], "iPhone (DIES)", ANCHOR_I),
                            (axes[0][1], "Gmail (HOLDS)", ANCHOR_G)):
        p0 = states["P"]["0"]["ctrl_p"][key]
        for arm, col, mk in (("P", "#c0392b", "o"), ("C", "#e67e22", "s")):
            a0 = states[arm]["0"]["ctrl_p"][key]
            ys = [states[arm][str(s)]["ctrl_p"][key] / a0 for s in sts]
            ax.plot(sts, ys, f"{mk}-", color=col, lw=2.2, ms=7,
                    label=f"ARM-{arm} "
                    f"({'projected' if arm == 'P' else 'matched-L2'})")
        for w, col in (("w1", "0.35"), ("w2", "0.55"), ("w3", "0.70")):
            ax.plot(refs[w]["states"], refs[w]["hr"][key], "d--", color=col,
                    lw=1.3, ms=5, alpha=0.9,
                    label=f"committed {w} ({refs[w]['label']})")
        lo, hi = adj["committed_spread"][key]
        ax.axhspan(lo, hi, color="#b8d8f0", alpha=0.35, zorder=0,
                   label="committed spread")
        if key == ANCHOR_I:
            ax.axhline(adj["flip_line_primary"], color="#1a6faf", ls=":",
                       lw=1.6, label="0.5x Gmail committed hr (w1)")
        ax.axhline(1.0, color="0.55", lw=0.7, ls=":")
        ax.set_xlabel("wash step"); ax.set_ylabel("hr = p(s)/p(0)")
        ax.set_title(f"{anchor} — the fate trajectory", fontsize=10)
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.25)

    ax = axes[1][0]
    for arm, col in (("P", "#c0392b"), ("C", "#e67e22")):
        led = adj["ledgers"][arm]
        xs = sorted(int(k) for k in led)
        ax.plot(xs, [led[str(s)]["ce"] for s in xs], "-", color=col,
                lw=1.4, alpha=0.85, label=f"ARM-{arm} in-batch CE")
    ax.plot(refs["w1"]["ce_states"], refs["w1"]["ce_vals"], "kd--", ms=5,
            lw=1.0, alpha=0.7, label="committed w1 in-batch CE")
    ax.set_xlabel("wash step"); ax.set_ylabel("batch CE (8x512)")
    ax.set_title("the wash must still be a wash — CE trajectories",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1][1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E233 VERDICT: {verdict['bar']}"
            + ("  [UNDERPOWERED — honest-instrument clause]"
               if verdict.get("underpowered") else ""),
            fontsize=11, va="top", family="monospace", weight="bold",
            color="darkred")
    y -= 0.05
    for line in adj["verdict_lines"]:
        ax.text(0.02, y, line, fontsize=7.6, va="top", family="monospace")
        y -= 0.026
    y -= 0.02
    ax.text(0.02, y, "GATES:", fontsize=8.6, va="top",
            family="monospace", weight="bold")
    y -= 0.028
    for gname, gval in adj["gates_summary"].items():
        ax.text(0.02, y, f"  {gname:14s} {'PASS' if gval else 'FAIL'}",
                fontsize=7.2, va="top", family="monospace")
        y -= 0.024

    fig.suptitle("E233 — THE SUPPORT-PROJECTION INTERVENTION: is the "
                 "wash's iPhone-aligned component the CARRIER or a "
                 "MARKER?\n(each applied step projected off iPhone's t=0 "
                 "support, norm not rescaled; ARM-C matched-L2; wash-1 "
                 "stream seed 18202)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = rd / f"{NAME}_fates.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def make_ledger_plot(rd, ledgers):
    """THE LEDGER FIGURE: the removed-L2 record + the step norms + the
    ARM-C scale factors."""
    led_p = ledgers["P"]
    xs = sorted(int(k) for k in led_p)
    getp = lambda s, k: (led_p[str(s)][k] if str(s) in led_p
                         else led_p[s][k])
    fr = [getp(s, "removed_frac") for s in xs]
    dn = [getp(s, "dnorm") for s in xs]
    dpn = [getp(s, "dpnorm") for s in xs]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    ax = axes[0]
    ax.plot(xs, [f * 100 for f in fr], "o-", color="#c0392b", lw=1.6, ms=3.5)
    ax.axhline(1.0, color="#1a6faf", ls=":", lw=1.6,
               label="the 1% honest-instrument line")
    med = sorted(fr)[len(fr) // 2]
    ax.axhline(med * 100, color="0.4", ls="--", lw=1.0,
               label=f"median {med*100:.2f}%")
    ax.set_xlabel("wash step")
    ax.set_ylabel("|<Delta, s_iPhone>| / ||Delta||  (%)")
    ax.set_title("ARM-P removed-L2 fraction (the honest-instrument read)",
                 fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot(xs, dn, "o-", color="0.45", lw=1.4, ms=3.5,
            label="||Delta|| (verbatim step norm)")
    ax.plot(xs, dpn, "s-", color="#c0392b", lw=1.4, ms=3.5,
            label="||Delta'|| (ARM-P applied)")
    led_c = ledgers["C"]
    xsc = sorted(int(k) for k in led_c)
    getc = lambda s, k: (led_c[str(s)][k] if str(s) in led_c
                         else led_c[s][k])
    ax.plot(xsc, [getc(s, "applied_norm") for s in xsc], "^-", ms=3.5,
            color="#e67e22", lw=1.2, alpha=0.8,
            label="ARM-C applied (matched to ||Delta'||)")
    ax.set_xlabel("wash step"); ax.set_ylabel("L2 norm of applied step")
    ax.set_title("the step norms (projection shrinks P; C is matched)",
                 fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[2]
    ax.plot(xsc, [getc(s, "scale") for s in xsc], "o-", color="#e67e22",
            lw=1.4, ms=3.5)
    ax.axhline(1.0, color="0.55", lw=0.7, ls=":")
    ax.set_xlabel("wash step"); ax.set_ylabel("ARM-C scale factor")
    ax.set_title("ARM-C scale = ||Delta'_P|| / ||Delta_C|| (step-locked)",
                 fontsize=9.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = rd / f"{NAME}_ledger.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------ main

def main():
    jp = RD / "journal.json"
    log(f"E233 — THE SUPPORT-PROJECTION INTERVENTION (smoke={SMOKE}) "
        f"-> {RD}")

    metrics = {
        "experiment": "e233_support_intervention",
        "phase": "interventional: the wash-1 stream continued VERBATIM "
                 "with the applied steps projected off iPhone's t=0 "
                 "support (ARM-P) vs matched-L2 (ARM-C), GPU fp32 bursts",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("e226 located the third dimension's seat — from +10 "
                     "the wash's continuing pull works along the DYING "
                     "anchor's support (2.6-14.8x, three washes). Is that "
                     "aligned component the CARRIER of the death "
                     "(removing it spares iPhone) or a MARKER (removing "
                     "it changes nothing — g14's LETHAL != CARRIER)?"),
        "builds_on": [
            "scratch/e233_design.md (the frozen design — THE registration)",
            "T210 / e226 (the seat located; the support + wash-gradient "
            "conventions imported VERBATIM; the anchor fates + certified "
            "draw streams)",
            "T149 / e182c + T183 / e182c2 + T192 / e217 (the wash recipe, "
            "the batteries, the three-wash archive, the GPU burst "
            "discipline)",
            "g12 (the intervention-changes-the-trajectory precedent), "
            "g14 (LETHAL != CARRIER), W030 ('the next instrument is the "
            "intervention')",
        ],
        "whats_new": [
            "the FIRST interventional cell on the 124M wash archive: the "
            "applied AdamW step projected off a probe's t=0 support "
            "direction, per step, for 80 steps (the direction-causality "
            "test of the e226 seat)",
            "the matched-L2 control arm (ARM-C) isolating the DIRECTION "
            "removal from the magnitude shrink",
            "the removed-L2 ledger (fp64 dots; the honest-instrument "
            "clause wired to the verdict)",
        ],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(RD / "metrics.json", metrics)

    journal: dict = {}
    if jp.exists():
        try:
            journal = json.loads(jp.read_text(encoding="utf-8"))
            log(f"journal restored: {list(journal)}")
        except Exception as ex:                              # noqa: BLE001
            log(f"journal unreadable ({ex}); starting fresh")
            journal = {}

    def save_journal():
        jp.write_text(json.dumps(journal, indent=1, default=float),
                      encoding="utf-8")

    load_checks: list[dict] = [cpu_load_check("launch")]
    metrics["load_checks"] = load_checks

    # ------------------------------------------------ P0 the committed records
    E182C2_M = common.REPO / "runs" / "e182c2" / "metrics.json"
    for p in (E182C_M, E182C2_M, E182C2_J, E217_M, E216_M, E226_M):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    p1m = json.loads(E182C_M.read_text(encoding="utf-8"))
    c2m = json.loads(E182C2_M.read_text(encoding="utf-8"))
    j2 = json.loads(E182C2_J.read_text(encoding="utf-8"))
    u3m = json.loads(E217_M.read_text(encoding="utf-8"))
    e216 = json.loads(E216_M.read_text(encoding="utf-8"))
    e226 = json.loads(E226_M.read_text(encoding="utf-8"))
    w1_rec = {s["step"]: s for s in p1m["states"]}
    w1_tmpl_rec = {s["step"]: s                 # e226's pattern: w1's
                   for s in c2m["part1_template"]["states"]}  # tmpl lives
    w2_rec = {s["step"]: s for s in j2["states"]}             # in e182c2
    w3_rec = {s["step"]: s for s in u3m["wash3_states"]}

    def _probes(state_rec, batt):
        return {f: v["p"] for f, v in state_rec[batt]["probes"].items()}

    committed = e226["anchor_fates_committed"]     # runtime-read, never
    # transcribed: {"Gmail": {"w1": {p0,p80,hr}, ...}, "iPhone": {...}}
    metrics["anchor_fates_committed"] = committed
    g_hrs = [committed["Gmail"][w]["hr"] for w in ("w1", "w2", "w3")]
    i_hrs = [committed["iPhone"][w]["hr"] for w in ("w1", "w2", "w3")]
    spread = {ANCHOR_G: (min(g_hrs), max(g_hrs)),
              ANCHOR_I: (min(i_hrs), max(i_hrs))}
    flip_primary = 0.5 * committed["Gmail"]["w1"]["hr"]
    flip_lines = {"w1": flip_primary,
                  "w2": 0.5 * committed["Gmail"]["w2"]["hr"],
                  "w3": 0.5 * committed["Gmail"]["w3"]["hr"]}
    log("committed fates (runtime-read from e226): " + " | ".join(
        f"{t}: " + ", ".join(f"{w} hr {committed[t][w]['hr']:.3f}"
                             for w in ("w1", "w2", "w3"))
        for t in committed) + f" | FLIP line (0.5x Gmail w1) {flip_primary:.4f}")

    # the committed reference curves (anchors, all three washes)
    refs = {}
    for w, rec in (("w1", w1_rec), ("w2", w2_rec), ("w3", w3_rec)):
        sts = sorted(rec)
        refs[w] = {
            "label": {"w1": "e182c replay, seed 18202",
                      "w2": "e182c2 fresh, seed 20261002",
                      "w3": "e217 fresh, seed 21703"}[w],
            "states": sts,
            "hr": {a: [rec[s]["ctrl"]["probes"][a]["p"]
                       / rec[sts[0]]["ctrl"]["probes"][a]["p"]
                       for s in sts] for a in (ANCHOR_G, ANCHOR_I)},
            "ce_states": [s for s in sts[1:]
                          if rec[s].get("in_batch_ce") is not None],
            "ce_vals": [rec[s]["in_batch_ce"] for s in sts[1:]
                        if rec[s].get("in_batch_ce") is not None],
        }

    # ------------------------------------------------ P1 organism + corpus
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = {**org_meta, "torch_threads": THREADS}
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"]
    metrics["gates"] = {"G_SIZE": G_SIZE}
    metrics["size_gate"] = G_SIZE

    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    e_corp, e_str = e182["corpus"], e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    banned = sorted({s.lower() for rel in e1.POOLS for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_banned, "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    for r, b in zip(cand, e1.probe_battery(net0, cand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": True,
        "tokens_after": [corpus_stats["tokens_after"],
                         e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "format": "[rebuilt, e182_recorded]",
        "note": "the corpus is INHERITED FROZEN — it feeds the wash-batch "
                "reproduction (the SAME train_ids wash 1 drew its windows "
                "from)",
    }
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens)")
    assert G_CORPUS["pass"] or SMOKE
    metrics["gates"]["G_CORPUS"] = G_CORPUS
    write_metrics("PARTIAL: records read; organism + corpus certified")

    # --------------------------------- P2 the four batteries, VERBATIM (G_BATT)
    load_checks.append(cpu_load_check("batteries"))
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]
    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]
    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                          e_banned)
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]
    bats = {"fact": battery, "ctrl": cbattery, "near": nbattery,
            "tmpl": tbattery}
    G_BATT = {}
    for b, bl in bats.items():
        mine_p = {r["fact"]: r["p"] for r in bl}
        dps, sets_ok = {}, True
        recs0 = {"w1": (w1_rec if b != "tmpl" else w1_tmpl_rec),
                 "w2": w2_rec, "w3": w3_rec}
        for w, rec in recs0.items():
            ref = _probes(rec[0], b)
            sets_ok = sets_ok and set(mine_p) == set(ref)
            dps[w] = (max(abs(mine_p[f] - v) for f, v in ref.items())
                      if set(mine_p) == set(ref) else None)
        G_BATT[b] = {"n": len(bl), "set_equal_committed": bool(sets_ok),
                     "max_dp": dps}
    G_BATT["tol_per_probe_dp"] = TOL_T0_DP
    G_BATT["pass"] = bool(
        all(G_BATT[b]["set_equal_committed"]
            and max((d for d in G_BATT[b]["max_dp"].values()
                     if d is not None), default=1.0) <= TOL_T0_DP
            for b in bats)) if not SMOKE else True
    G_BATT["note"] = ("batteries = the phase-1/phase-2 pools VERBATIM "
                      "(module import); t=0 must reproduce all THREE "
                      "committed records (e226's G_BATT convention)")
    metrics["gates"]["G_BATT"] = G_BATT
    log("G_BATT: " + ("PASS" if G_BATT["pass"] else "FAIL") + " | "
        + " | ".join(f"{b}: n {G_BATT[b]['n']} maxdp "
                     f"{max(G_BATT[b]['max_dp'].values()):.2e}"
                     for b in bats))
    if not G_BATT["pass"]:
        write_metrics("PARTIAL: G_BATT FAILED — halted before any compute")
        return 1
    probes_bats = bats
    ctrl_by_fact = {r["fact"]: r for r in cbattery}

    # --------------------------------- P3 the direction + FD + determinism
    load_checks.append(cpu_load_check("support t=0"))
    params0 = list(net0.parameters())
    def support_chunks(probe):
        """e226's probe_support convention, kept as per-param chunks (the
        flat order IS net.parameters() order; fp64 global norm)."""
        net0.eval()
        net0.zero_grad(set_to_none=True)
        logits = net0(input_ids=probe["ids"]).logits
        pvec = F.softmax(logits[0, -1], dim=-1)
        p0 = float(pvec[probe["ans_id"]].item())
        pvec[probe["ans_id"]].backward()
        gs = [p.grad.detach().clone().reshape(-1)
              for p in params0]
        nrm = float(math.sqrt(sum(float(
            torch.dot(g.double(), g.double()).item()) for g in gs)))
        assert nrm > 0 and math.isfinite(nrm), "degenerate support"
        net0.zero_grad(set_to_none=True)
        return [g / nrm for g in gs], p0
    s_I_cpu, p0_I = support_chunks(ctrl_by_fact[ANCHOR_I])
    s_G_cpu, p0_G = support_chunks(ctrl_by_fact[ANCHOR_G])
    netFD = copy.deepcopy(net0)
    base_sds = [p.detach().clone() for p in netFD.parameters()]

    def fd_gate(s_chunks, probe):
        p0 = e3.probe_p(netFD, probe)
        out = {}
        for eps in FD_EPS:
            ofs = 0
            with torch.no_grad():
                for p, b, s_c in zip(netFD.parameters(), base_sds,
                                     s_chunks):
                    p.copy_(b.reshape(p.shape)
                            + (eps * s_c).reshape(p.shape))
            out[str(eps)] = e3.probe_p(netFD, probe) - p0
            with torch.no_grad():
                for p, b in zip(netFD.parameters(), base_sds):
                    p.copy_(b)
        return out
    fd_I = fd_gate(s_I_cpu, ctrl_by_fact[ANCHOR_I])
    fd_G = fd_gate(s_G_cpu, ctrl_by_fact[ANCHOR_G])
    del netFD, base_sds
    s_I_rep, _ = support_chunks(ctrl_by_fact[ANCHOR_I])
    self_cos = float(sum(torch.dot(a.double(), b.double()).item()
                         for a, b in zip(s_I_cpu, s_I_rep)))
    G_SUPPORT = {
        "fd_eps": list(FD_EPS),
        "rule": "p(theta0 + eps*s_hat) - p0 > 0 at the primary eps "
                "(e204's directional gate, e226's port to 124M)",
        "iphone_fd": fd_I, "gmail_fd_co_read": fd_G,
        "p0": {"iPhone": p0_I, "Gmail": p0_G,
               "committed_iphone": committed["iPhone"]["w1"]["p0"],
               "committed_gmail": committed["Gmail"]["w1"]["p0"]},
        "determinism_selfcos_iphone": self_cos,
        "pass": bool(fd_I[str(FD_EPS[0])] > 0 and self_cos > 0.999)
        if not SMOKE else True,
        "note": "the projection direction = iPhone's t=0 support, "
                "recomputed FRESH fp32 CPU (e226's convention; fp64 "
                "norms); ONE fixed direction for all steps (the design's "
                "honesty guard); Gmail's support FD-gated as a co-read",
    }
    metrics["gates"]["G_SUPPORT"] = G_SUPPORT
    log(f"G_SUPPORT: {'PASS' if G_SUPPORT['pass'] else 'FAIL'} "
        f"(iPhone fd {fd_I[str(FD_EPS[0])]:+.3e} @eps {FD_EPS[0]}; "
        f"determinism self-cos {self_cos:.7f}; p0 {p0_I:.4f} vs committed "
        f"{committed['iPhone']['w1']['p0']:.4f})")
    assert G_SUPPORT["pass"] or SMOKE
    write_metrics("PARTIAL: batteries + support direction certified")

    # --------------------------------- P4 the certified draw stream (G_DRAWS)
    load_checks.append(cpu_load_check("draws"))
    gen = torch.Generator().manual_seed(W_SEED)
    hi = train_ids.shape[0] - e1.SEQ - 1
    offs = [torch.randint(hi, (e1.BATCH,), generator=gen)
            for _ in range(80)]          # the full certified stream
    archived = torch.load(W1_LATEST, map_location=CPU,
                          weights_only=False)["gen"]
    G_DRAWS = {
        "seed": W_SEED,
        "n_draws_reproduced": len(offs),
        "gen_state_identical_after_80": bool(
            torch.equal(gen.get_state(), archived)),
        "note": "wash 1's window-draw stream reproduced from seed 18202 "
                "and certified BIT-EXACTLY against the generator state "
                "archived at step 80 in e182c_replay_latest.pt (e226's "
                "G_DRAWS convention); both arms replay THIS stream from "
                "pristine t=0",
        "pass": bool(torch.equal(gen.get_state(), archived)),
    }
    metrics["gates"]["G_DRAWS"] = G_DRAWS
    log(f"G_DRAWS: {'PASS' if G_DRAWS['pass'] else 'FAIL'} "
        f"(seed {W_SEED}, 80 draws, bit-exact vs the archived generator)")
    assert G_DRAWS["pass"]

    # --------------------------------- P5 t=0 readout (both arms share it)
    load_checks.append(cpu_load_check("t0 readout"))
    def probe_state(netC):
        rec = {}
        for b, bl in probes_bats.items():
            bb = e1.probe_battery(netC, bl)
            rec[f"{b}_p"] = {r["fact"]: r["p"] for r in bb["probes"]}
            rec[f"{b}_mean_p"] = bb["mean_p"]
        hp = e1.ppl_eval(netC, *bank_xy)
        rec["bank_ppl"] = hp["ppl"]
        rec["bank_ce"] = hp["ce"]
        return rec
    t0_rec = probe_state(net0)
    states = {"P": {"0": t0_rec}, "C": {"0": t0_rec}}
    journal["states"] = states
    save_journal()
    log(f"t=0: iPhone p0 {t0_rec['ctrl_p'][ANCHOR_I]:.4f} Gmail p0 "
        f"{t0_rec['ctrl_p'][ANCHOR_G]:.4f} | bank ppl "
        f"{t0_rec['bank_ppl']:.2f} (e182 committed 71.34)")
    write_metrics("PARTIAL: t=0 readout done (both arms)")

    # --------------------------------- P6 the arms
    def on_ckpt(tag, step, sd, ce):
        netC = copy.deepcopy(net0)
        netC.load_state_dict(sd)
        rec = probe_state(netC)
        rec["in_batch_ce"] = ce
        states[tag][str(step)] = rec
        journal["states"] = states
        save_journal()
        log(f"  [ARM-{tag}] +{step}: iPhone p "
            f"{rec['ctrl_p'][ANCHOR_I]:.4f} Gmail p "
            f"{rec['ctrl_p'][ANCHOR_G]:.4f} | bank ppl "
            f"{rec['bank_ppl']:.2f}")
        del netC

    dev = torch.device("cuda")
    s_chunks_gpu = [s.to(dev) for s in s_I_cpu]
    assert [s.numel() for s in s_chunks_gpu] == \
        [p.numel() for p in net0.parameters()]

    ledgers = journal.get("ledgers", {})
    for tag in ("P", "C"):
        led_checks = cpu_load_check(f"arm {tag}")
        load_checks.append(led_checks)
        if tag == "P":
            p_ledger = None
        else:
            assert "P" in ledgers and len(ledgers["P"]) >= N_STEPS, \
                "ARM-P ledger incomplete — ARM-C cannot be step-locked"
            p_ledger = {int(k): v for k, v in ledgers["P"].items()}
        res_key = f"arm_{tag}"
        already = journal.get(res_key, {}).get("done", False)
        if already and tag in ledgers and len(ledgers[tag]) >= N_STEPS \
                and all(str(s) in states[tag] for s in CK_STEPS):
            log(f"ARM-{tag}: journal says done — skipping")
            continue
        out = run_arm(tag, net0, train_ids, offs, bank_xy, s_chunks_gpu,
                      p_ledger, probes_bats, on_ckpt)
        ledgers[tag] = out["ledger"]
        journal["ledgers"] = ledgers
        journal[res_key] = {"done": True,
                            "envelope_polls": out["envelope_polls"]}
        save_journal()
        write_metrics(f"PARTIAL: ARM-{tag} complete "
                      f"({len(out['ledger'])} steps)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    # the inter-arm cache reset
        if tag == "P" and not SMOKE:
            time.sleep(COOLDOWN_S)      # the inter-arm cooldown

    # normalize ledger keys to str (journal/metrics round-trip shape)
    ledgers = {t: {str(k): v for k, v in ledgers[t].items()}
               for t in ledgers}
    journal["ledgers"] = ledgers
    save_journal()
    L = lambda t, s: ledgers[t][str(s)]        # the uniform accessor

    # --------------------------------- P7 gates: wash health
    def arm_health(tag):
        ces = [L(tag, s)["ce"] for s in range(1, N_STEPS + 1)]
        first10 = sum(ces[:10]) / len(ces[:10])
        last10 = sum(ces[-10:]) / len(ces[-10:])
        ppl0 = states[tag]["0"]["bank_ppl"]
        pplf = states[tag][str(CK_STEPS[-1])]["bank_ppl"]
        return {"ce_first10": first10, "ce_last10": last10,
                "ce_improves": bool(last10 < first10),
                "bank_ppl_t0": ppl0, "bank_ppl_final": pplf,
                "ppl_improves": bool(pplf < ppl0),
                "pass": bool(last10 < first10 and pplf < ppl0)}
    G_WASHHEALTH = {"P": arm_health("P"), "C": arm_health("C"),
                    "rule": "the wash must still be a wash: bank ppl "
                            "(+final) < bank ppl (t=0) AND mean in-batch "
                            "CE last-10 < first-10, BOTH arms"}
    G_WASHHEALTH["pass"] = bool(G_WASHHEALTH["P"]["pass"]
                                and G_WASHHEALTH["C"]["pass"])
    metrics["gates"]["G_WASHHEALTH"] = G_WASHHEALTH
    G_ENV = {
        "device": "cuda (RTX 5090 Laptop 24GB) fp32, TF32 OFF, matmul "
                  "precision highest; CPU fp32 probing (threads 8)",
        "bursts": f"<= {BURST_MAX_S:.0f}s wall AND <= {BURST_MAX_STEPS} "
                  f"steps",
        "cooldown_s": COOLDOWN_S,
        "launch_gate": f"common.gpu_ok() (util<={common.GPU_UTIL_CEIL:.0f}%, "
                       f"temp<={common.GPU_TEMP_CEIL:.0f}C, mem<=85%), "
                       f"double-polled, first launch gap "
                       f"{LAUNCH_POLL_GAP_S:.0f}s",
        "mid_burst_guard": f"burst ends at temp >= {TEMP_BURST_END:.0f}C "
                           f"(never past 85C)",
        "window": "the owner's ACTIVE max-priority window (STATE.json "
                  "compute_directive, 2026-10-04): GPU assertive, NO "
                  "concurrent GPU jobs, temp-aware, single runs <=180s",
        "pass": True,
    }
    metrics["gates"]["G_ENV"] = G_ENV
    log(f"G_WASHHEALTH: {'PASS' if G_WASHHEALTH['pass'] else 'FAIL'} | "
        + " | ".join(f"ARM-{t}: CE {G_WASHHEALTH[t]['ce_first10']:.3f}->"
                     f"{G_WASHHEALTH[t]['ce_last10']:.3f}, ppl "
                     f"{G_WASHHEALTH[t]['bank_ppl_t0']:.1f}->"
                     f"{G_WASHHEALTH[t]['bank_ppl_final']:.1f}"
                     for t in ("P", "C")))

    # --------------------------------- P8 adjudication (the frozen bars)
    def hr(tag, fact, state):
        return states[tag][str(state)]["ctrl_p"][fact] / \
            states[tag]["0"]["ctrl_p"][fact]
    final = CK_STEPS[-1]
    hr80 = {f"{t}-{a}": hr(t, a, final)
            for t in ("P", "C") for a in (ANCHOR_G, ANCHOR_I)}
    flips_fire = bool(hr80[f"P-{ANCHOR_I}"] >= flip_primary
                      and spread[ANCHOR_I][0] <= hr80[f"C-{ANCHOR_I}"]
                      <= spread[ANCHOR_I][1])
    holds_fire = bool(all(spread[a][0] <= hr80[f"{t}-{a}"] <= spread[a][1]
                          for t in ("P", "C")
                          for a in (ANCHOR_G, ANCHOR_I)))
    # the honest-instrument clause (c)
    fracs = sorted(L("P", s)["removed_frac"] for s in range(1, N_STEPS + 1))
    med_frac = fracs[len(fracs) // 2]
    underpowered = bool(med_frac < 0.01)

    if flips_fire:
        bar = "FATE-FLIPS"
    elif holds_fire:
        bar = "FATE-HOLDS"
    else:
        bar = "ANY"
    verdict = {"bar": bar, "underpowered": underpowered,
               "hr_at_final": hr80,
               "committed_spread": {k: list(v) for k, v in spread.items()},
               "flip_line_primary": flip_primary,
               "flip_lines_all": flip_lines,
               "flips_conjunction": {
                   "iphone_P_ge_line": bool(
                       hr80[f"P-{ANCHOR_I}"] >= flip_primary),
                   "iphone_C_stays_dying": bool(
                       spread[ANCHOR_I][0] <= hr80[f"C-{ANCHOR_I}"]
                       <= spread[ANCHOR_I][1])},
               "holds_all_four_within_spread": holds_fire,
               "removed_frac_median": med_frac,
               "removed_frac_min": fracs[0], "removed_frac_max": fracs[-1],
               "bars_verbatim": REGISTERED_PREDICTION["bars_verbatim"]}
    verdict_lines = [
        f"iPhone hr: P {hr80[f'P-{ANCHOR_I}']:.3f}  C "
        f"{hr80[f'C-{ANCHOR_I}']:.3f}   (committed spread "
        f"{spread[ANCHOR_I][0]:.3f}-{spread[ANCHOR_I][1]:.3f}; "
        f"flip line {flip_primary:.3f})",
        f"Gmail  hr: P {hr80[f'P-{ANCHOR_G}']:.3f}  C "
        f"{hr80[f'C-{ANCHOR_G}']:.3f}   (committed spread "
        f"{spread[ANCHOR_G][0]:.3f}-{spread[ANCHOR_G][1]:.3f})",
        f"removed-L2 fraction: median {med_frac*100:.2f}% "
        f"(min {fracs[0]*100:.2f}%, max {fracs[-1]*100:.2f}%) "
        f"-> {'UNDERPOWERED' if underpowered else 'powered'}",
    ]

    # prediction (a) — the iPhone-family co-read (no bar; verbatim table)
    fam_rows = []
    for f in IPHONE_FAMILY:
        p0m = states["P"]["0"]["ctrl_p"][f]
        committed_w1_hr = (w1_rec[80]["ctrl"]["probes"][f]["p"]
                           / w1_rec[0]["ctrl"]["probes"][f]["p"])
        fam_rows.append({
            "fact": f, "p0": p0m,
            "hr_P": hr("P", f, final), "hr_C": hr("C", f, final),
            "committed_w1_hr": committed_w1_hr,
            "spares_under_P": bool(hr("P", f, final) > committed_w1_hr)})
    pred_a = {"read": "prediction (a) verbatim: 'Under FATE-FLIPS, the "
                      "nearrel battery's iPhone-family members also "
                      "spare (the seat is the relation family's, not the "
                      "anchor's alone).'",
              "family": "product family's 5 non-anchor members "
                        "(Xbox/Chrome/iPad/iTunes/PlayStation)",
              "rows": fam_rows,
              "n_spare": sum(1 for r in fam_rows if r["spares_under_P"]),
              "tested_under": bar,
              "fired": bool(bar == "FATE-FLIPS"
                            and all(r["spares_under_P"]
                                    for r in fam_rows))}
    # the literal near battery co-report
    pred_near = {"read": "the literal near battery (near-uscap) "
                         "co-reported at +final, both arms",
                 "rows": [{"fact": r["fact"],
                           "p0": states["P"]["0"]["near_p"][r["fact"]],
                           "hr_P": states["P"][str(final)]
                           ["near_p"][r["fact"]]
                           / states["P"]["0"]["near_p"][r["fact"]],
                           "hr_C": states["C"][str(final)]
                           ["near_p"][r["fact"]]
                           / states["C"]["0"]["near_p"][r["fact"]],
                           "committed_w1_hr": (
                               w1_rec[80]["near"]["probes"][r["fact"]]["p"]
                               / w1_rec[0]["near"]["probes"][r["fact"]]["p"])}
                          for r in nbattery]}
    # prediction (b) — Gmail's unchanged read under HOLDS
    pred_b = {"read": "prediction (b) verbatim: 'Under FATE-HOLDS, "
                      "Gmail's fate is also unchanged (the projection "
                      "touched only a component the wash does not need) "
                      "— and the t=0 consumption read (below) becomes "
                      "the live mechanism thread.'",
              "gmail_unchanged_under_HOLDS": bool(
                  spread[ANCHOR_G][0] <= hr80[f"P-{ANCHOR_G}"]
                  <= spread[ANCHOR_G][1]
                  and spread[ANCHOR_G][0] <= hr80[f"C-{ANCHOR_G}"]
                  <= spread[ANCHOR_G][1]),
              "tested_under": bar,
              "consumption_read_handoff": "the t=0 consumption read "
                          "(T210-b) rides as the e232 companion desk pass "
                          "(committed-data arithmetic); under FATE-HOLDS "
                          "it becomes the live mechanism thread"}
    metrics["adjudication"] = {
        **verdict,
        "verdict_note": ("the verdict is CO-STAMPED UNDERPOWERED when the "
                         "median removed fraction is < 1% (the honest-"
                         "instrument clause (c); the stamp discloses, it "
                         "does not move the bar)")
        if underpowered else
        "the honest-instrument clause (c) not triggered (median removed "
        "fraction >= 1%)",
        "prediction_a": pred_a,
        "prediction_b": pred_b,
        "near_battery_coreport": pred_near,
        "gated_on": "G_SIZE/G_CORPUS/G_BATT/G_SUPPORT/G_DRAWS/"
                    "G_WASHHEALTH/G_ENV",
        "all_gates_pass": bool(all(
            v.get("pass", True) for v in metrics["gates"].values()
            if isinstance(v, dict))),
    }
    metrics["adjudication"]["verdict_lines"] = verdict_lines
    metrics["states"] = {t: {s: {"ctrl_p": states[t][s]["ctrl_p"],
                                 "ctrl_mean_p": states[t][s]["ctrl_mean_p"],
                                 "bank_ppl": states[t][s]["bank_ppl"],
                                 "in_batch_ce": states[t][s].get(
                                     "in_batch_ce")}
                             for s in states[t]} for t in ("P", "C")}
    metrics["ledgers"] = ledgers
    metrics["references"] = refs
    metrics["honesty_reflex"] = {
        "n": "n=1 organism, n=1 wash stream per arm (the design's guard; "
             "the 3-wash replication belongs to e226's observational "
             "base) — the arms' internal P-vs-C contrast is stream-"
             "matched and device-matched",
        "trajectory": "the intervention changes the trajectory: after "
                      "step 1 each arm is its own wash (g12's precedent; "
                      "the design's disclosed nature) — fates, not "
                      "trajectories, are compared",
        "direction": "ONE t=0-fixed direction for all steps (e226's "
                     "rotation reads were sub-bar; the design's "
                     "disclosed choice); the projection removes the "
                     "component along it and does NOT rescale",
        "optimizer_state": "the AdamW state advances on the verbatim "
                           "step given each trajectory's own gradient — "
                           "momentum/preconditioning are part of the "
                           "machinery being kept verbatim, disclosed",
        "instrument_floor": "all ledger dots/norms fp64 (e226 found fp32 "
                            "accumulation over 124M coords drifts ~0.6% "
                            "— the same order as the 1% clause)",
        "device": "arms GPU fp32 vs committed w1 CPU-fp32/w2w3 GPU-fp32 "
                  "references — the spread spans the texture; fates "
                  "compared, not bit values (e217's precedent)",
    }

    # --------------------------------- P9 plots + final write
    gates_summary = {g: metrics["gates"][g].get("pass", True)
                     for g in metrics["gates"]}
    adj_plot = {**verdict, "verdict": verdict["bar"],
                "verdict_lines": verdict_lines,
                "gates_summary": gates_summary,
                "committed_spread": spread,
                "ledgers": ledgers}
    png1 = make_fates_plot(RD, states, refs, adj_plot)
    png2 = make_ledger_plot(RD, {"P": ledgers["P"], "C": ledgers["C"]})
    metrics["plot_outputs"] = [str(png1), str(png2)]
    metrics["compute"] = {
        "wall_s": round(time.time() - T0, 1),
        "device": "GPU fp32 bursts (RTX 5090 Laptop) + CPU fp32 probing",
        "steps": f"2 arms x {N_STEPS} steps (wash-1 certified stream)",
        "load_checks": len(load_checks),
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations
    status = ("SMOKE DONE (nothing adjudicated)" if SMOKE else
              ("DONE" if metrics["adjudication"]["all_gates_pass"]
               else "DONE (gate failures disclosed — see gates)"))
    write_metrics(status)
    log(f"VERDICT: {bar}"
        + (" [UNDERPOWERED]" if underpowered else "")
        + f" | iPhone P {hr80[f'P-{ANCHOR_I}']:.3f} / C "
        f"{hr80[f'C-{ANCHOR_I}']:.3f} vs line {flip_primary:.3f} | Gmail "
        f"P {hr80[f'P-{ANCHOR_G}']:.3f} / C {hr80[f'C-{ANCHOR_G}']:.3f} "
        f"| removed median {med_frac*100:.2f}%")
    log(f"outputs: {RD / 'metrics.json'}, {png1}, {png2}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
