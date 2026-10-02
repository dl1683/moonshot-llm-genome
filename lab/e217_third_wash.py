"""E217 — THE THIRD WASH DRAW (T185's named thread).

WHY: e213's census (runs/e213; T185) mapped the 124M path-independence at
n=2 wash draws: the +50 mid-dose is a STATE FUNCTION (every battery within
+-9%; the template battery's three-decimal match), the +10 shallows are
path-typed (wash 2 erodes earlier, every non-floor ratio > 1.1), and the
+80 deep regime PARTIALLY MEAN-REVERTS (the deepest wash-1 declines read
shallower on wash 2 — near 0.857, tmpl 0.804 as ratios; a sparing
correlate, Spearman -0.60 on the four batteries). But n=2 draws cannot
separate "the deep drift is growing" from "wash 1's depth was the
outlier". THE THIRD DRAW decides it.

THE CELL: a THIRD fresh 5e-5 wash draw at 124M — a NEW registered stream
seed (21703), the e182c2 part-2 machinery VERBATIM (GPT-2 pinned, the
frozen corpus, the fact battery + the control battery + the near-related
+ the template battery; t=0/+10/+50/+80 in short owner-envelope GPU
bursts, the batteries read CPU fp32). Then THE THREE-WASH CENSUS:
  * the path-independence ratios now n=3 (pairwise, anchored on wash 1 —
    e213's instrument): does the +50 mid-dose state function TIGHTEN, and
    does the +80 deep drift GROW or MEAN-REVERT?
  * the relational signature's cross-wash reliability at n=3 (the family
    hold-ratios' spread; e215's typology runtime-read, hr = p(+80)/p_0).

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any wash
compute; adjudicate against exactly this; no bar shopping):
  - MID-DOSE-TIGHTENS: "fires if the +50 ratios' spread shrinks with the
    third draw (max |ratio-1| <= 0.15 on all four batteries) — the
    mid-dose state function licensed at n=3."
  - DEEP-DRIFT-GROWS: "fires if the +80 drift widens (the deepest
    batteries' ratios move further from 1 in the same direction) — the
    deep-regime path-sensitivity is real and growing; the three-regime
    picture final."
  - DEEP-MEAN-REVERTS: "fires if the +80 drift shrinks or flips —
    wash 1's depth was the outlier; the state function deepens its
    claim."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * declines: decl_b^w(s) = 1 - R_b^w(s)/R_b^w(0), R = battery mean
    p(answer first token); wash-1/wash-2 declines RUNTIME-READ from
    runs/e213/metrics.json (census.declines — never transcribed); wash-3
    computed here, same CPU fp32 probe instrument.
  * ratios: e213's instrument VERBATIM, anchored on wash 1:
    r21_b(s) = decl_w2/decl_w1, r31_b(s) = decl_w3/decl_w1, and the
    third pair r32_b(s) = decl_w3/decl_w2 co-reported. Denominator
    guard: the bar is read only if decl_w1 > 0.02 (e213's DECL_FLOOR) at
    that state; else not read, disclosed (not expected at +50/+80: all
    eight wash-1/2 declines there are >= 0.275).
  * MID-DOSE-TIGHTENS (at +50): fires iff max |ratio-1| over the three
    pairwise ratios {r21, r31, r32} <= 0.15 for EVERY battery (all
    four). "Spread shrinks" is co-reported (max |r-1| with vs without
    the third draw); the FIRE is the parenthetical test.
  * DEEP bars (at +80): "the deepest batteries" := the TWO batteries
    with the largest wash-1 +80 declines in the runtime-read e213 record
    (near 0.766, tmpl 0.561 — frozen from the committed record, never
    re-derived from wash 3). Per battery b:
      d2 = |r21_b - 1|, d3 = |r31_b - 1|, same_sign = (r21_b-1)*(r31_b-1) > 0
      GROWS_b   := same_sign AND d3 > d2   (further from 1, same direction)
      REVERTS_b := (not same_sign) OR d3 < d2   (flip or shrink)
      FLAT_b    := exact tie (d3 == d2, not same-sign-grown) — GRADED
    DEEP-DRIFT-GROWS fires iff GROWS on BOTH deepest batteries;
    DEEP-MEAN-REVERTS fires iff REVERTS on BOTH; anything else (mixed or
    FLAT) -> neither deep bar -> GRADED (the tables verbatim).
  * GRADED co-fires with any partial (the deep pair mixed, or a bar not
    read under its denominator guard).
  * RELATIONAL (co-report, no bar): hr_w(probe) = p_w(+80)/p_0; e215's
    6-family typology + per-probe hr_w1/hr_w2 RUNTIME-READ from
    runs/e215/metrics.json; hr_w3 from THIS wash's +80 probes over THIS
    t=0 (same pristine organism; identity gated by G_BATT). Family
    statistic = MEDIAN hr (e215's instrument); spread = max-min across
    the three washes; pooled per-probe Spearman pairs (w1xw2 reproduces
    e215's 0.939; w3xw1, w3xw2 new); HOLD/COLLAPSE/MID classes at n=3.
  * Adjudication is GATED on the verification gates below; if any fails:
    VERIFICATION-FAILED, curves reported, no bar read.

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_SIZE — 124,439,808 params <= 500M with e182's stated reason.
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to e182's
    recorded filter stats (40001 lines / 664 dropped / 1093972 chars /
    331770 tokens / train 319481 / bank 24x512, banned list identical).
    The corpus is INHERITED FROZEN — the third draw changes ONLY the
    stream seed.
  * G_BATT — the four batteries rebuilt VERBATIM (module import) must
    reproduce e213's committed t=0 census readings (same pristine
    organism, same prompts, same CPU fp32 probe path): same kept sets,
    per-probe |dp| <= 0.010 on all four batteries; co-checked against
    e215's per-probe p0.
  * G_TMPL — the template battery identity: kept set equal to the
    committed e213/e215 tmpl battery; contamination scans zero (e182c2's
    own gate, inherited via the module import).
  * G_STATES_PROV — the wash-1/wash-2 state archives (e182c_s*.pt,
    e182c2_fresh_s*.pt) still on disk with sizes matching e213's
    committed inventory (the w1/w2 declines are runtime-read from the
    e213 record; e213 itself verified the archives by re-probe, max dp
    3.34e-06 — that stamp is inherited, not re-run here).
  * G_PPL — the held-out bank perplexity read at every wash-3 state and
    IMPROVES (the health reference).
  * G_ENV — the owner envelope: every GPU launch double-polled
    (util <= 20% AND temp <= 70C, >= 5 s apart), every poll logged to
    the run log AND runs/_envelope_log.jsonl; bursts <= 75 s wall AND
    <= 40 steps; cooldown >= 180 s between bursts (CPU probing counts
    toward it); no GPU outside bursts; CPU load-checked, threads 4
    (the CPU desk is shared — e213's precedent; the ~1e-6 probe texture
    vs the records is absorbed by the tolerances).

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the openness is
the point. n=3 draws is still texture, not law; a tightening mid-dose
does not prove state-functionality (three draws can coincide on a loose
band); a growing or reverting deep drift at n=3 does not settle the deep
regime (the drift may live in the last 10% of the response). The device
texture stands: wash 1's states are e182c's CPU fp32 REPLAY of e182's
GPU wash (its G_REPLAY stamp <= 0.001 fact mean_p dev); washes 2 and 3
train GPU fp32 (TF32 OFF) — the census compares declines (patterns),
never bit values. Probes are CPU fp32 everywhere (the phase-1
instrument). nearrel n=3 (item-level noise the 19-20 item batteries do
not carry). The relational co-report inherits e215's typology as
committed (hand-registered families — the family label is an
instrument, not a finding).

COMPUTE ENVELOPE (the owner envelope, permanent): the lab is the LOWEST
priority — GPU launch only on double-polled util <= 20% / temp <= 70C;
bursts <= 75 s (the 80-step wash split at the checkpoints: ~10 / 40 /
30 steps); cooldown >= 180 s between bursts (CPU probing between bursts
counts toward it); every poll logged. Progressive PARTIAL metrics +
resumable journal after every state (the standing disruption rule).

PROVENANCE: the organism, the frozen corpus filter+verify, the wash
recipe (AdamW (0.9,0.95) wd 0.1 constant lr 5e-5, clip 1.0, batch 8 x
ctx 512, CPU-generator window draws — the device-independent
bit-identical-draw discipline), the batteries (fact/control/nearrel +
the reversed-form template) and probe_battery/select_battery/ppl_eval
are the e182c2 part-2 machinery VERBATIM via module import
(lab/e182c_forgetting_control.py -> lab/e182c2_template.py, which
themselves inherit lab/e182_gpt2_wash.py). The w1/w2 census values and
the relational typology are RUNTIME-READ from runs/e213/metrics.json
and runs/e215/metrics.json (never transcribed). Builds on: e213/T185
(the census + the drift), e182c2/T183 (the fresh-draw machinery + the
free find), e215/T189 (the relational typology), e182c/T149 + e182/T123
(the saved-state discipline + the parent wash). NEW: the third stream
seed 21703 (registered here before compute), the three-wash census
(pairwise ratios at n=3), the deep-drift grow/revert adjudication, the
relational signature's n=3 reliability table.

Run:  cd lab && python e217_third_wash.py   (E217_SMOKE=1: 2-step
      wash, grid {1,2}, own smoke dir, nothing adjudicated)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch                                          # noqa: E402

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import now_iso, run_dir, save_json          # noqa: E402

import e182c_forgetting_control as e1                  # noqa: E402 — the phase-1 machinery, VERBATIM
import e182c2_template as e2                           # noqa: E402 — the part-2 machinery (template battery + fresh-draw discipline), VERBATIM

torch.set_num_threads(4)          # e213's shared-CPU precedent (the CPU desk is busy; disclosed)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E217_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e217_smoke" if SMOKE else "e217"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen constants (registered BEFORE compute) ------------------------
FRESH_SEED = 21703              # the THIRD stream seed (w1: 18202, w2: 20261002)
FRESH_STEPS = 2 if SMOKE else 80
FRESH_CK: tuple[int, ...] = (1, 2) if SMOKE else (10, 50, 80)
CENSUS_STATES: tuple[int, ...] = (1, 2) if SMOKE else (10, 50, 80)
MID_STATE = 2 if SMOKE else 50      # the MID-DOSE bar's state
DEEP_STATE = 2 if SMOKE else 80     # the DEEP bars' state
BATTERIES = ("fact", "ctrl", "near", "tmpl")

# the owner envelope (stricter than common's gpu_ok launch gate)
LAUNCH_UTIL, LAUNCH_TEMP = 20.0, 70.0
POLL_GAP_S = 5.0
BURST_MAX_S = 75.0
BURST_MAX_STEPS = 40
COOLDOWN_S = 180.0

# ---- registered bar constants (frozen) ----------------------------------------
RATIO_TOL_MID = 0.15            # MID-DOSE-TIGHTENS: max |ratio-1| (all 3 pairs, all 4 batteries)
DECL_FLOOR = 0.02               # e213's floor (denominator guard)

# ---- registered verification tolerances (frozen) ------------------------------
TOL_PROBE_DP = 0.010            # per-probe t=0 dp vs e213's committed census t=0
TOL_P0_E215_DP = 0.010          # co-check vs e215's per-probe p0

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "MID-DOSE-TIGHTENS": "fires if the +50 ratios' spread shrinks with "
            "the third draw (max |ratio-1| <= 0.15 on all four batteries) "
            "— the mid-dose state function licensed at n=3.",
        "DEEP-DRIFT-GROWS": "fires if the +80 drift widens (the deepest "
            "batteries' ratios move further from 1 in the same direction) "
            "— the deep-regime path-sensitivity is real and growing; the "
            "three-regime picture final.",
        "DEEP-MEAN-REVERTS": "fires if the +80 drift shrinks or flips — "
            "wash 1's depth was the outlier; the state function deepens "
            "its claim.",
        "GRADED": "any partial — the tables verbatim.",
    },
    "lean": "T185's map (mid-dose tight at +-9%, deep partially mean-reverting "
        "with the deepest batteries shallower on wash 2) leans "
        "MID-DOSE-TIGHTENS + DEEP-MEAN-REVERTS; but n=2 draws, the near "
        "battery n=3, and the CPU-replay-vs-GPU device asymmetry between the "
        "wash-1 arm and the wash-2/3 arms — nothing guaranteed; the openness "
        "is the point",
    "operationalizations": "decl_b^w(s)=1-R_b^w(s)/R_b^w(0); w1/w2 declines "
        "runtime-read from runs/e213/metrics.json, w3 computed here (CPU "
        "fp32 probes); ratios anchored on wash 1 (e213's instrument): "
        "r21=decl_w2/decl_w1, r31=decl_w3/decl_w1, r32=decl_w3/decl_w2; "
        "denominator guard decl_w1>0.02 (DECL_FLOOR); MID at +50: max "
        "|ratio-1| over {r21,r31,r32} <= 0.15 for EVERY battery; DEEP at "
        "+80: deepest batteries := the two largest wash-1 +80 declines in "
        "the e213 record (near 0.766, tmpl 0.561 — frozen); per battery "
        "GROWS := same_sign(r21-1, r31-1) AND |r31-1| > |r21-1|; REVERTS "
        ":= sign-flip OR |r31-1| < |r21-1|; FLAT := exact tie; "
        "DEEP-DRIFT-GROWS iff GROWS on both, DEEP-MEAN-REVERTS iff REVERTS "
        "on both, else GRADED; GRADED co-fires with any partial; relational "
        "co-report (no bar): hr_w = p_w(+80)/p_0, family = MEDIAN hr, "
        "e215 typology runtime-read, spread = max-min across washes, "
        "pooled Spearman pairs; adjudication gated on "
        "G_SIZE/G_CORPUS/G_BATT/G_TMPL/G_STATES_PROV/G_PPL",
    "registration": "bars frozen VERBATIM from the dispatch brief (QUEUE row "
        "DISPATCHED 15:47Z); the third stream seed 21703 and these "
        "operationalizations registered in this file and committed BEFORE "
        "any wash compute; no bar shopping",
}

deviations: list[str] = [
    "The w1/w2 census values are RUNTIME-READ from runs/e213/metrics.json "
    "(never transcribed); the wash-1/wash-2 state archives are NOT re-probed "
    "here — e213 already verified them by re-probe (max dp 3.34e-06, its "
    "G_STATES stamp, inherited); this cell checks provenance by the on-disk "
    "size/mtime inventory against e213's committed record (G_STATES_PROV).",
    "torch threads 4 (the shared-CPU precedent; e213 ran 4 where the records' "
    "probes ran 8) — CPU fp32 reduction-order texture may shift per-probe p "
    "in the ~1e-6..1e-4 range; the G_BATT tolerance (0.010) absorbs it; "
    "declines are compared at the 3-decimal precision the bars name.",
    "Device texture: wash 1's states are e182c's CPU fp32 REPLAY of e182's "
    "GPU wash; washes 2 and 3 train GPU fp32 (TF32 OFF) — the census "
    "compares declines (patterns), never bit values.",
    "The deepest batteries ({near, tmpl}) are FROZEN from the committed e213 "
    "record (the two largest wash-1 +80 declines) — never re-derived from "
    "wash 3 (no bar shopping).",
    "The relational co-report inherits e215's hand-registered 6-family "
    "typology VERBATIM (runtime-read) — the family label is an instrument; "
    "hr_w3 uses THIS cell's t=0 as the denominator (identity gated by "
    "G_BATT vs e213's committed t=0 and co-checked vs e215's p0).",
    "Fresh-draw per-state weights saved (runs/checkpoints/"
    "e217_fresh_s{N}.pt + resumable e217_fresh_latest.pt) so no future "
    "phase re-runs this wash.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: 2-step fresh wash, grid {1,2}, own smoke dir and log, "
    "nothing adjudicated or verified.",
    "PLOT-ONLY RE-PASS (the e182c2 precedent): the first full pass wrote "
    "complete metrics + journal off a clean 80-step wash (455.2s, 3 "
    "owner-envelope launch cycles, all polls FREE) and then both figures "
    "crashed on two plot-only bugs (a mis-keyed dict access "
    "'max_abs_dev' vs 'max_abs_dev_all_pairs'; the spearman block "
    "iterating the dict's 'note' string through a float format). The fix "
    "is plot-only + recording-only (plus the G_ENV carry-forward of the "
    "wash pass's envelope record); main() re-ran SELF-RESUMING off the "
    "frozen journal — zero wash recompute, zero GPU, the census numbers "
    "bit-identical (deterministic arithmetic on the same journal values "
    "+ the same runtime-read committed records).",
]


# ------------------------------------------------------------------ envelope

def gpu_poll(tag: str) -> dict:
    s = common.gpu_status()
    ok = s["util"] <= LAUNCH_UTIL and s["temp"] <= LAUNCH_TEMP
    common._log_envelope_poll(f"{NAME}:{tag}", s["util"], s["temp"], ok)
    log(f"  [gpu:{tag}] util {s['util']:.0f}% temp {s['temp']:.0f}C "
        f"mem {s['mem_used']:.0f}/{s['mem_total']:.0f}MB "
        f"power {s['power']:.1f}W -> {'FREE' if ok else 'BUSY'}")
    return {"poll": s, "ok": bool(ok)}


def gpu_free_double_poll(tag: str) -> tuple[bool, list[dict]]:
    """The owner-envelope launch gate: two polls >=5 s apart, BOTH
    util <= 20% AND temp <= 70C. Every poll logged (run log + the
    envelope audit log)."""
    p1 = gpu_poll(f"{tag}#1")
    time.sleep(POLL_GAP_S)
    p2 = gpu_poll(f"{tag}#2")
    return bool(p1["ok"] and p2["ok"]), [p1["poll"], p2["poll"]]


def wait_for_free(tag: str, max_wait_s: float = 3600.0) -> list[dict]:
    """Pause-and-wait (never migrate): poll until the envelope opens."""
    t_start = time.time()
    while True:
        ok, polls = gpu_free_double_poll(tag)
        if ok:
            return polls
        if time.time() - t_start > max_wait_s:
            raise RuntimeError(f"GPU never freed within {max_wait_s}s "
                               f"for {tag}")
        log(f"  [gpu:{tag}] waiting 30s for a free window "
            f"(owner envelope: util<={LAUNCH_UTIL:.0f}% "
            f"temp<={LAUNCH_TEMP:.0f}C)")
        time.sleep(30.0)


def cpu_load_check(tag: str) -> dict:
    try:
        import psutil
        pct = psutil.cpu_percent(interval=0.6)
    except Exception:                                   # noqa: BLE001
        pct = None
    rec = {"tag": tag, "cpu_percent": pct, "tool": "psutil" if pct is not None else "unavailable"}
    log(f"  [cpu:{tag}] load {pct if pct is None else round(pct, 1)}%")
    return rec


# --------------------------------------------------------------- the fresh wash

def fresh_wash(net0, train_ids, ckpt_steps, on_ckpt, resume=None):
    """The THIRD draw of e182's frozen 5e-5 wash (the e182c2 part-2
    machinery verbatim), GPU fp32 (TF32 OFF) in owner-envelope bursts:
    AdamW (0.9,0.95) wd 0.1 constant lr 5e-5, clip 1.0, full-token CE,
    batch 8 x ctx 512; per step BATCH window offsets from the CPU
    generator seeded FRESH_SEED (e182's device-independent
    bit-identical-draw discipline — the ONLY delta from washes 1/2).
    Bursts: <= BURST_MAX_STEPS steps AND <= BURST_MAX_S wall, launched
    only on double-polled free windows; >= COOLDOWN_S between bursts;
    checkpoint steps end their burst. Resumable state saved after every
    burst; per-state weight archives saved at checkpoints."""
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=e1.LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(FRESH_SEED)
    hi = train_ids.shape[0] - e1.SEQ - 1
    step = 0
    ckpt_dir = common.REPO / "runs" / "checkpoints"
    latest = ckpt_dir / f"{NAME}_fresh_latest.pt"
    if resume is not None:
        net.load_state_dict(resume["model"])
        net.to(dev)
        opt.load_state_dict(resume["opt"])
        e2.opt_state_dev_fix(opt, dev)
        gen.set_state(resume["gen"])
        step = int(resume["step"])
        log(f"fresh wash: RESUMED from step {step} (generator state "
            f"continues bit-identically)")
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    envelope_polls: list[dict] = []
    while step < n_steps:
        burst_id = step + 1
        envelope_polls += wait_for_free(f"burst{burst_id}")
        t_burst = time.time()
        n_burst, hit_ckpt = 0, None
        last_ce = None
        while step < n_steps and n_burst < BURST_MAX_STEPS \
                and time.time() - t_burst < BURST_MAX_S:
            step += 1
            n_burst += 1
            off = torch.randint(hi, (e1.BATCH,), generator=gen)
            x = torch.stack([train_ids[o: o + e1.SEQ] for o in off]).to(dev)
            y = torch.stack([train_ids[o + 1: o + 1 + e1.SEQ]
                             for o in off]).to(dev)
            logits = net(input_ids=x).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            last_ce = float(loss.item())
            if step % 10 == 0:
                log(f"  [fresh] s{step:3d}/{n_steps} corpus CE "
                    f"{last_ce:.4f} ({(time.time() - t_burst):.1f}s into "
                    f"burst)")
            if step in ckpt_set:
                hit_ckpt = step
                break
        t_end = time.time()
        log(f"  [fresh] burst done: {n_burst} steps in "
            f"{t_end - t_burst:.1f}s (<= {BURST_MAX_S:.0f}s cap), now at "
            f"s{step}" + (f", checkpoint +{hit_ckpt}" if hit_ckpt else ""))
        sd = {k: v.detach().to("cpu", torch.float32).clone()
              for k, v in net.state_dict().items()}
        opt_sd = e2._opt_state_to_cpu(opt.state_dict())
        torch.save({"model": sd, "opt": opt_sd, "gen": gen.get_state(),
                    "step": step,
                    "meta": {"experiment": NAME, "seed": FRESH_SEED,
                             "lr": e1.LR}}, latest)
        if hit_ckpt is not None:
            torch.save({"model": sd,
                        "meta": {"experiment": NAME, "step": hit_ckpt,
                                 "lr": e1.LR, "seed": FRESH_SEED,
                                 "desc": f"openai-community/gpt2@"
                                 f"{e1.MODEL_REV} E217 THIRD draw "
                                 f"(seed {FRESH_SEED}) GPU fp32, step "
                                 f"{hit_ckpt}", "base": e1.MODEL_REPO,
                                 "revision": e1.MODEL_REV}},
                       ckpt_dir / f"{NAME}_fresh_s{hit_ckpt}.pt")
        if hit_ckpt is not None:
            on_ckpt(hit_ckpt, sd, last_ce)
        del sd
        # cooldown: >= COOLDOWN_S since burst end (probing time counts)
        remain = COOLDOWN_S - (time.time() - t_end)
        if remain > 0 and step < n_steps:
            log(f"  [thermal] cooldown {remain:.0f}s (owner envelope "
                f">={COOLDOWN_S:.0f}s between bursts)")
            time.sleep(max(remain, 0.0))
    return step, envelope_polls


# ------------------------------------------------------------------ helpers

def spearman(xs, ys) -> float:
    """Spearman rank correlation with average-rank tie handling."""
    def _ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 0 else float("nan")


def hold_class(hr: float) -> str:
    return "HOLD" if hr >= 0.5 else ("COLLAPSE" if hr < 0.3 else "MID")


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ------------------------------------------------------------------ plots

def plot_three_wash(rd, census, adj, retention, mid_state, deep_state):
    """Figure 1: the three-wash census — retention curves, the +50
    mid-dose ratios, the +80 deep drift, the adjudication tables."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "tab:purple"}
    wash_style = {"w1": (":", "o", "wash 1 (seed 18202, CPU-replay arm)"),
                  "w2": ("--", "s", "wash 2 (seed 20261002, GPU fp32)"),
                  "w3": ("-", "D", "wash 3 (seed 21703, GPU fp32 — THIS CELL)")}
    steps = sorted(retention["w3"])
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    ax = axes[0, 0]
    for b in BATTERIES:
        for w, (ls, mk, lab) in wash_style.items():
            ys = [retention[w][s][b] / retention[w][0][b] for s in steps]
            ax.plot(steps, ys, ls, marker=mk, ms=6 if w != "w3" else 7,
                    lw=1.6 if w != "w3" else 2.4, color=cols[b],
                    alpha=0.55 if w == "w1" else (0.75 if w == "w2" else 1.0),
                    label=f"{b} {lab.split(' (')[0]}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("wash steps (shared states)")
    ax.set_ylabel("retention R(s)/R(0)")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, ncol=2, loc="lower left")
    ax.set_title("THE THREE-WASH CENSUS — retention per battery per draw",
                 fontsize=10)

    ax = axes[0, 1]
    x = range(len(BATTERIES))
    w = 0.26
    fmt = lambda v: f"{v:7.3f}".strip() if v is not None else "n/a"
    for i, (wk, col) in enumerate((("w1", "gray"), ("w2", "tab:cyan"),
                                   ("w3", "tab:orange"))):
        vals = [census["declines"][wk][str(mid_state)][b] for b in BATTERIES]
        ax.bar([xx + (i - 1) * w for xx in x], vals, w, color=col,
               alpha=0.85 if wk == "w3" else 0.65,
               label={"w1": "wash 1", "w2": "wash 2", "w3": "wash 3"}[wk])
    for i, b in enumerate(BATTERIES):
        rr = census["pairwise"][str(mid_state)][b]
        ax.text(i, max(census["declines"][w][str(mid_state)][b]
                       for w in ("w1", "w2", "w3")) + 0.012,
                f"r21 {fmt(rr['r21'])}\nr31 {fmt(rr['r31'])}\nr32 {fmt(rr['r32'])}",
                ha="center", fontsize=6.4, family="monospace")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(BATTERIES)
    ax.set_ylabel(f"decline at +{mid_state}")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(census["declines"][w][str(mid_state)][b]
                           for w in ("w1", "w2", "w3") for b in BATTERIES) * 1.28, 0.05))
    m = adj["mid_dose"]
    max_dev_plot = m["max_abs_dev_all_pairs"]
    max_dev_plot_txt = f"{max_dev_plot:.3f}" if max_dev_plot is not None else "n/a"
    ax.set_title(f"MID-DOSE at +{mid_state}: max|r-1| {max_dev_plot_txt} "
                 f"(tol {RATIO_TOL_MID}) -> "
                 f"{'FIRES' if m['fires'] else 'does not fire'}", fontsize=9.5)

    ax = axes[1, 0]
    for i, (wk, col) in enumerate((("w1", "gray"), ("w2", "tab:cyan"),
                                   ("w3", "tab:orange"))):
        vals = [census["declines"][wk][str(deep_state)][b] for b in BATTERIES]
        ax.bar([xx + (i - 1) * w for xx in x], vals, w, color=col,
               alpha=0.85 if wk == "w3" else 0.65,
               label={"w1": "wash 1", "w2": "wash 2", "w3": "wash 3"}[wk])
    for i, b in enumerate(BATTERIES):
        rr = census["pairwise"][str(deep_state)][b]
        star = " *" if b in adj["deep_drift"]["deepest_batteries"] else ""
        ax.text(i, max(census["declines"][w][str(deep_state)][b]
                       for w in ("w1", "w2", "w3")) + 0.012,
                f"r21 {fmt(rr['r21'])}{star}\nr31 {fmt(rr['r31'])}{star}",
                ha="center", fontsize=6.6, family="monospace")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}\n(*deepest)" if b in adj["deep_drift"]["deepest_batteries"]
                        else b for b in BATTERIES])
    ax.set_ylabel(f"decline at +{deep_state}")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(census["declines"][w][str(deep_state)][b]
                           for w in ("w1", "w2", "w3") for b in BATTERIES) * 1.28, 0.05))
    d = adj["deep_drift"]
    ax.set_title(f"DEEP at +{deep_state}: deepest {d['deepest_batteries']} -> "
                 f"{d['fired']}", fontsize=9.5)

    ax = axes[1, 1]
    ax.axis("off")
    y = 0.995
    ax.text(0.02, y, "GATES:", fontsize=8.4, va="top", family="monospace",
            weight="bold")
    y -= 0.030
    for g, val in adj["gates_summary"].items():
        ax.text(0.02, y, f"  {g:14s} {'PASS' if val else 'FAIL'}",
                fontsize=6.8, va="top", family="monospace")
        y -= 0.021
    y -= 0.012
    ax.text(0.02, y, "THREE-WASH RATIOS (anchored on wash 1; r_ij = decl_i/decl_j):",
            fontsize=8.0, va="top", family="monospace", weight="bold")
    y -= 0.028
    for st in census["states"]:
        if st in (mid_state, deep_state):
            ax.text(0.02, y, f"+{st}:", fontsize=7.0, va="top",
                    family="monospace", weight="bold")
            y -= 0.024
            for b in BATTERIES:
                rr = census["pairwise"][str(st)][b]
                ax.text(0.02, y,
                        f" {b:5s} r21 {fmt(rr['r21']):>6s} r31 {fmt(rr['r31']):>6s}"
                        f" r32 {fmt(rr['r32']):>6s}",
                        fontsize=6.2, va="top", family="monospace")
                y -= 0.022
            y -= 0.005
    y -= 0.012
    ax.text(0.02, y, f"FIRED: {', '.join(adj['fired']) or 'none'}",
            fontsize=8.6, va="top", family="monospace", weight="bold",
            color="darkred")
    y -= 0.036
    for wd in textwrap.wrap(adj["clause"], width=78, break_long_words=False):
        ax.text(0.02, y, f" {wd}", fontsize=6.2, va="top", family="monospace")
        y -= 0.021

    fig.suptitle(f"E217 — THE THIRD WASH DRAW (seed {FRESH_SEED}) -> "
                 f"{adj['verdict_composite']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "three_wash_census.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def plot_relational(rd, rel):
    """Figure 2: the relational signature at n=3 — family medians of the
    hold ratio across the three washes + the per-probe reliability."""
    fams = rel["family_order"]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.2))

    ax = axes[0]
    x = range(len(fams))
    w = 0.26
    for i, (wk, col) in enumerate((("w1", "gray"), ("w2", "tab:cyan"),
                                   ("w3", "tab:orange"))):
        vals = [rel["family_medians"][f][wk] for f in fams]
        ax.bar([xx + (i - 1) * w for xx in x], vals, w, color=col,
               alpha=0.85 if wk == "w3" else 0.65,
               label={"w1": "wash 1", "w2": "wash 2", "w3": "wash 3"}[wk])
    for i, f in enumerate(fams):
        sp = rel["family_spread"][f]["spread_max_min"]
        ax.text(i, max(rel["family_medians"][f][w] for w in ("w1", "w2", "w3"))
                + 0.015, f"spread\n{sp:.3f}", ha="center", fontsize=6.3,
                family="monospace")
    ax.axhline(0.5, color="seagreen", ls=":", lw=1.2, label="HOLD band (0.5)")
    ax.axhline(0.3, color="darkred", ls=":", lw=1.2, label="COLLAPSE band (0.3)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("family MEDIAN hold ratio hr = p(+80)/p0")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7.5)
    ax.set_title("THE RELATIONAL SIGNATURE AT n=3 — family medians per draw",
                 fontsize=9.5)

    ax = axes[1]
    xs1 = [p["hr_w1"] for p in rel["per_probe"]]
    xs2 = [p["hr_w2"] for p in rel["per_probe"]]
    ys = [p["hr_w3"] for p in rel["per_probe"]]
    ax.scatter(xs2, ys, s=14, color="tab:cyan", alpha=0.6,
               label=f"hr_w3 vs hr_w2 (rho {rel['spearman']['w3xw2']:.3f})")
    ax.scatter(xs1, ys, s=14, color="gray", alpha=0.5,
               label=f"hr_w3 vs hr_w1 (rho {rel['spearman']['w3xw1']:.3f})")
    ax.scatter(xs2, xs1, s=8, color="dimgray", alpha=0.25, marker="x",
               label=f"hr_w2 vs hr_w1 (e215's 0.939; rho "
                     f"{rel['spearman']['w2xw1']:.3f})")
    ax.axhline(0.5, color="seagreen", ls=":", lw=1.0)
    ax.axhline(0.3, color="darkred", ls=":", lw=1.0)
    ax.set_xlabel("hr_w1 / hr_w2 (committed records)")
    ax.set_ylabel("hr_w3 (THIS CELL)")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("per-probe reliability across draws (54 probes)", fontsize=9.5)

    ax = axes[2]
    ax.axis("off")
    y = 0.97
    ax.text(0.03, y, "FAMILY TABLE (median hr per wash; spread = max-min):",
            fontsize=8.8, va="top", family="monospace", weight="bold")
    y -= 0.030
    hdr = (f"  {'family':14s} {'n':>3s} {'w1':>7s} {'w2':>7s} {'w3':>7s} "
           f"{'spread':>7s} {'class3':>14s}")
    ax.text(0.03, y, hdr, fontsize=7.0, va="top", family="monospace")
    y -= 0.024
    for f in fams:
        r = rel["family_medians"][f]
        sp = rel["family_spread"][f]
        row = (f"  {f:14s} {sp['n']:3d} {r['w1']:7.3f} {r['w2']:7.3f} "
               f"{r['w3']:7.3f} {sp['spread_max_min']:7.3f} "
               f"{rel['family_class3_w3'][f]:>14s}")
        ax.text(0.03, y, row, fontsize=7.0, va="top", family="monospace")
        y -= 0.024
    y -= 0.014
    ax.text(0.03, y, "pooled per-probe Spearman:", fontsize=8.4, va="top",
            family="monospace", weight="bold")
    y -= 0.026
    for k in ("w2xw1", "w3xw1", "w3xw2"):
        v = rel["spearman"][k]
        ax.text(0.03, y, f"  {k:6s} rho {v:.3f}", fontsize=7.4, va="top",
                family="monospace")
        y -= 0.023
    y -= 0.012
    ax.text(0.03, y, "hold classes at n=3 (54 probes):", fontsize=8.4,
            va="top", family="monospace", weight="bold")
    y -= 0.026
    for k, v in rel["class3_counts"].items():
        ax.text(0.03, y, f"  {k:22s} {v:3d}", fontsize=7.4, va="top",
                family="monospace")
        y -= 0.023
    y -= 0.010
    for wd in textwrap.wrap(rel["note"], width=88, break_long_words=False):
        ax.text(0.03, y, wd, fontsize=6.6, va="top", family="monospace",
                color="dimgray")
        y -= 0.020

    fig.suptitle("E217 — THE RELATIONAL SIGNATURE'S CROSS-WASH RELIABILITY "
                 "AT n=3 (e215's typology runtime-read; co-report, no bar)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = rd / "relational_signature_n3.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"E217 — THE THIRD WASH DRAW (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e217_third_wash",
        "phase": "the third fresh 5e-5 wash draw + the three-wash census "
                 "+ the relational signature at n=3",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": ("bars frozen VERBATIM from the dispatch brief "
                         "(QUEUE DISPATCHED 15:47Z); the third stream seed "
                         "21703 + operationalizations registered and "
                         "committed BEFORE any wash compute; no bar shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("T185's named thread: with a THIRD independent wash "
                     "draw, does the +80 deep drift GROW (deep-regime "
                     "path-sensitivity real and growing) or MEAN-REVERT "
                     "(wash 1's depth was the outlier)? and does the +50 "
                     "mid-dose state function tighten at n=3?"),
        "builds_on": ["e213 / T185 (the two-wash census + the drift)",
                      "e182c2 / T183 (the fresh-draw machinery + the free "
                      "find)",
                      "e215 / T189 (the relational typology)",
                      "e182c / T149 + e182 / T123 (the saved-state "
                      "discipline + the parent wash)"],
        "whats_new": ["the third stream seed 21703 (n=3 wash draws)",
                      "the three-wash census (pairwise ratios anchored on "
                      "wash 1)",
                      "the deep-drift grow/revert adjudication",
                      "the relational signature's n=3 reliability table"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    load_checks = [cpu_load_check("launch")]

    # ---------------------------------------------------- P0 the records
    e213_path = common.REPO / "runs" / "e213" / "metrics.json"
    e215_path = common.REPO / "runs" / "e215" / "metrics.json"
    for p in (e213_path, e215_path):
        if not p.exists():
            log(f"FATAL: record missing: {p}")
            return 1
    e213m = json.loads(e213_path.read_text(encoding="utf-8"))
    e215m = json.loads(e215_path.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    cens = e213m["census"]
    decl_rec = cens["declines"]            # w1/w2 declines, runtime-read
    states_rec = e213m["census_states"]    # w1/w2 per-state battery records
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e182["gates"]["G_STR"]["banned"]
    # retention curves for w1/w2 from the committed census states
    retention = {"w1": {}, "w2": {}, "w3": {}}
    for s in states_rec:
        wk = s["wash"] if s["wash"] != "t0" else "w1"    # t0 shared row
        if s["wash"] == "t0":
            for wkk in ("w1", "w2"):
                retention[wkk][s["step"]] = {b: s[b]["mean_p"] for b in BATTERIES}
        else:
            retention[s["wash"]][s["step"]] = {b: s[b]["mean_p"] for b in BATTERIES}
    if not SMOKE:
        for wkk in ("w1", "w2"):
            for s_ in (0, *CENSUS_STATES):
                assert s_ in retention[wkk], f"e213 record lacks {wkk} +{s_}"

    # ---------------------------------------------------- P1 the organism
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = org_meta
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"], f"size envelope exceeded: {G_SIZE}"
    metrics["size_gate"] = G_SIZE

    # ------------------------------------------- P2 the frozen corpus (G_CORPUS)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    banned = sorted({s.lower() for rel in e1.POOLS for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_banned, "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    base_cand = e1.probe_battery(net0, cand)
    for r, b in zip(cand, base_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    kept_facts, sel = e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, bank_xy, filtered, G_STR, G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": True,
        "lines_total": [G_STR["lines_total"], e_str["lines_total"]],
        "lines_dropped": [G_STR["lines_dropped"], e_str["lines_dropped"]],
        "chars_after": [corpus_stats["chars_after"], e_corp["chars_after"]],
        "tokens_after": [corpus_stats["tokens_after"],
                         e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "bank_windows": [corpus_stats["bank_windows"],
                         e_corp["bank_windows"]],
        "format": "[rebuilt, e182_recorded]",
        "note": "the corpus is INHERITED FROZEN (the wash corpus cannot be "
                "re-filtered without changing the wash); the third draw "
                "changes ONLY the stream seed",
    }
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"]
        and corpus_stats["bank_windows"] == e_corp["bank_windows"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens; e182 "
        f"{e_corp['tokens_after']})")
    assert G_CORPUS["pass"] or SMOKE, f"corpus rebuild diverged: {G_CORPUS}"

    # --------------------------------- P3 the four batteries (G_BATT + G_TMPL)
    ccand, cdropped = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    cbase = e1.probe_battery(net0, ccand)
    for r, b in zip(ccand, cbase["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    ctrl_kept, _ = e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]

    ncand, ndropped = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    nbase = e1.probe_battery(net0, ncand)
    for r, b in zip(ncand, nbase["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    near_kept, _ = e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]

    tcand, tdropped = e2.build_tmpl_candidates(tok, filtered.lower(),
                                               train_ids, e_banned)
    tbase = e1.probe_battery(net0, tcand)
    for r, b in zip(tcand, tbase["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    tmpl_kept, tsel = e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]

    batteries = {"fact": battery, "ctrl": cbattery,
                 "near": nbattery, "tmpl": tbattery}

    # G_BATT: kept sets + t=0 probes vs e213's committed census t=0
    t0_rec = next(s for s in states_rec if s["wash"] == "t0")
    G_BATT = {"tol_per_probe_dp": TOL_PROBE_DP,
              "note": "the four batteries are the e182c2 part-2 pools "
                      "VERBATIM (module import); t=0 must reproduce e213's "
                      "committed census t=0 (same pristine organism, same "
                      "prompts, same CPU fp32 probe path)"}
    for bname, rows in batteries.items():
        p0p = t0_rec[bname]["probes"]
        dps = [abs(r["p"] - p0p[r["fact"]]["p"]) for r in rows
               if r["fact"] in p0p]
        G_BATT[bname] = {
            "kept_set_equal": bool({r["fact"] for r in rows} == set(p0p)),
            "n": len(dps),
            "max_per_probe_dp": max(dps) if dps else None,
        }
    # co-check vs e215's per-probe p0
    e215_assign = {r["fact"]: r for r in e215m["typology"]["assignment"]}
    e215_dps = [abs(r["p"] - e215_assign[r["fact"]]["p0"])
                for rows in batteries.values() for r in rows
                if r["fact"] in e215_assign]
    G_BATT["e215_p0_cocheck_max_dp"] = max(e215_dps) if e215_dps else None
    G_BATT["tol_e215_dp"] = TOL_P0_E215_DP
    G_BATT["pass"] = bool(
        all(G_BATT[b] and G_BATT[b]["kept_set_equal"]
            and G_BATT[b]["max_per_probe_dp"] <= TOL_PROBE_DP
            for b in BATTERIES)
        and (e215_dps and max(e215_dps) <= TOL_P0_E215_DP)) \
        if not SMOKE else True
    log(f"G_BATT: kept sets + t=0 vs e213 census record -> "
        f"{'PASS' if G_BATT['pass'] else 'FAIL'} "
        + " ".join(f"{b} dp {G_BATT[b]['max_per_probe_dp']:.2e}"
                   for b in BATTERIES)
        + f"; e215 p0 cocheck {G_BATT['e215_p0_cocheck_max_dp']:.2e}")

    # G_TMPL: template identity vs the committed record
    G_TMPL = {
        "kept_set_equal_vs_e213_t0":
            bool({r["fact"] for r in tbattery}
                 == set(t0_rec["tmpl"]["probes"])),
        "n": len(tbattery),
        "gate": tsel,
        "contamination_scans_zero": True,
        "pool_frozen_in_script": True,
        "pass": None,
    }
    G_TMPL["pass"] = bool(G_TMPL["kept_set_equal_vs_e213_t0"]
                          and len(tbattery) >= e1.BATTRY_FLOOR) \
        if not SMOKE else True
    log(f"G_TMPL: template battery n={len(tbattery)}, kept set vs e213 t0 "
        f"{'equal' if G_TMPL['kept_set_equal_vs_e213_t0'] else 'DIVERGED'}")

    # G_STATES_PROV: w1/w2 archives on disk, sizes vs e213's inventory
    inv213 = e213m["inventory"]
    prov = {"wash1_files": {}, "wash2_files": {}, "note":
            "w1/w2 declines runtime-read from the e213 record; e213 "
            "verified the archives by re-probe (max dp 3.34e-06) — that "
            "stamp inherited; here: existence + size match vs e213's "
            "committed inventory"}
    ok_prov = True
    for wk, key in (("wash1_files", "wash1_files"),
                    ("wash2_files", "wash2_files")):
        for st, rec in inv213[key].items():
            p = Path(rec["path"])
            cur = {"path": str(p), "exists": p.exists(),
                   "size_bytes": p.stat().st_size if p.exists() else None,
                   "recorded_size_bytes": rec["size_bytes"]}
            cur["size_match"] = bool(p.exists()
                                     and p.stat().st_size == rec["size_bytes"])
            ok_prov = ok_prov and cur["size_match"]
            prov[wk][st] = cur
    G_STATES_PROV = {**prov, "all_exist_and_match": bool(ok_prov),
                     "pass": bool(ok_prov) if not SMOKE else True}
    log(f"G_STATES_PROV: w1/w2 archives on disk + sizes vs e213 inventory -> "
        f"{'PASS' if G_STATES_PROV['pass'] else 'FAIL'}")
    metrics["inventory"] = {"prior_washes": prov}

    metrics["gates"] = {"G_SIZE": G_SIZE, "G_CORPUS": G_CORPUS,
                        "G_BATT": G_BATT, "G_TMPL": G_TMPL,
                        "G_STATES_PROV": G_STATES_PROV}

    # form-matching record (Rule 12)
    means = {b: sum(r["p"] for r in rows) / len(rows)
             for b, rows in batteries.items()}
    form_match = (
        "FORM-MATCHING: all four batteries are the e182c2 part-2 pools "
        "VERBATIM (module import; identity gated by G_BATT against e213's "
        "committed t=0 census record): 2-shot rotating leave-self-out "
        "cloze, single-token answers, gate (top1 p>=0.8)|(top5 p>=0.5), "
        f"cap 20; fact n={len(battery)} (R0 {means['fact']:.3f}), ctrl "
        f"n={len(cbattery)} (R0 {means['ctrl']:.3f}), near n="
        f"{len(nbattery)} (R0 {means['near']:.3f}), tmpl n="
        f"{len(tbattery)} (R0 {means['tmpl']:.3f}); one instrument, one "
        "pristine baseline, declines per battery relative; the three washes "
        "share the organism, the corpus, the recipe — the stream seed is "
        "the only delta.")
    metrics["form_matching"] = form_match
    log(form_match)
    write_metrics("PARTIAL: batteries built; the third draw pending")

    # ---------------------------------------------- P4 the third wash draw
    latest_ck = (common.REPO / "runs" / "checkpoints"
                 / f"{NAME}_fresh_latest.pt")
    p3_states = []
    resume = None
    if jp.exists():
        try:
            p3_states = json.loads(jp.read_text(encoding="utf-8"))["states"]
            log(f"journal: {len(p3_states)} states restored")
        except Exception as e:                          # noqa: BLE001
            log(f"journal unreadable ({e}); fresh")
            p3_states = []
    if latest_ck.exists():
        try:
            resume = torch.load(latest_ck, map_location=CPU,
                                weights_only=False)
            assert max(s["step"] for s in p3_states) == resume["step"], \
                "journal/checkpoint step mismatch"
        except Exception as e:                          # noqa: BLE001
            log(f"resume failed ({e}); fresh wash from scratch")
            resume = None
            p3_states = [s for s in p3_states if s["step"] == 0]
    metrics["wash3_states"] = p3_states   # self-contained on every pass

    def bat_rec(b):
        return {"mean_p": b["mean_p"], "frac_top1": b["frac_top1"],
                "frac_top5": b["frac_top5"],
                "probes": {r["fact"]: {"p": r["p"], "rank": r["rank"]}
                           for r in b["probes"]}}

    def state_rec(step, probes, hp, ce=None):
        return {"step": step, **{b: bat_rec(probes[b]) for b in BATTERIES},
                "ppl": hp["ppl"], "ce": hp["ce"], "in_batch_ce": ce}

    if 0 not in {s["step"] for s in p3_states}:
        load_checks.append(cpu_load_check("t0"))
        probes0 = {b: e1.probe_battery(net0, rows)
                   for b, rows in batteries.items()}
        hp = e1.ppl_eval(net0, *bank_xy)
        p3_states.append(state_rec(0, probes0, hp))
        p3_states.sort(key=lambda s: s["step"])
        jp.write_text(json.dumps({"states": p3_states}, indent=1),
                      encoding="utf-8")
        log(f"w3 t=0: " + " | ".join(f"{b} {probes0[b]['mean_p']:.4f}"
                                     for b in BATTERIES)
            + f" | ppl {hp['ppl']:.2f}")
        metrics["wash3_states"] = p3_states
        write_metrics("PARTIAL: w3 t=0 read; the wash pending")

    base_f = {b: p3_states[0][b]["mean_p"] for b in BATTERIES}

    def on_ckpt(step, sd, ce):
        load_checks.append(cpu_load_check(f"s{step}"))
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        probes = {b: e1.probe_battery(evl, batteries[b]) for b in BATTERIES}
        hp = e1.ppl_eval(evl, *bank_xy)
        del evl
        rec = state_rec(step, probes, hp, ce)
        p3_states.append(rec)
        p3_states.sort(key=lambda s: s["step"])
        jp.write_text(json.dumps({"states": p3_states}, indent=1),
                      encoding="utf-8")
        log(f"  W3 CKPT +{step:3d}: " + " | ".join(
            f"{b} {probes[b]['mean_p']:.4f} (decl "
            f"{1 - probes[b]['mean_p'] / base_f[b]:.3f})" for b in BATTERIES)
            + f" | ppl {hp['ppl']:.2f} | in-batch CE {ce:.4f}")
        metrics["wash3_states"] = p3_states
        write_metrics(f"PARTIAL: the third draw through +{step}")

    todo = tuple(s for s in FRESH_CK
                 if s > max(s2["step"] for s2 in p3_states))
    envelope_poll_log = []
    if todo:
        log(f"THE THIRD DRAW: lr {e1.LR}, {todo[-1]} steps, batch "
            f"{e1.BATCH} x ctx {e1.SEQ}, AdamW (0.9,0.95) wd 0.1 clip 1.0, "
            f"stream seed {FRESH_SEED} (washes 1/2: 18202 / 20261002), "
            f"checkpoints +{list(todo)}, GPU fp32 bursts "
            f"<={BURST_MAX_STEPS} steps/<={BURST_MAX_S:.0f}s, cooldown "
            f">={COOLDOWN_S:.0f}s, launches double-polled "
            f"util<={LAUNCH_UTIL:.0f}%/temp<={LAUNCH_TEMP:.0f}C")
        fin, envelope_poll_log = fresh_wash(net0, train_ids, todo,
                                            on_ckpt, resume=resume)
        log(f"the third draw finished at step {fin}")

    G_PPL = {"rule": "bank ppl (the SAME frozen held-out tail bank) read "
                     "at every wash-3 state",
             "ppl_curve": {str(s["step"]): s["ppl"] for s in p3_states},
             "ppl_improves": bool(p3_states[-1]["ppl"] < p3_states[0]["ppl"]),
             "pass": bool(p3_states[-1]["ppl"] < p3_states[0]["ppl"])}
    metrics["gates"]["G_PPL"] = G_PPL
    G_ENV = {"launch_gate": f"double-poll util<={LAUNCH_UTIL:.0f}% AND "
                            f"temp<={LAUNCH_TEMP:.0f}C, {POLL_GAP_S:.0f}s "
                            f"apart",
             "burst_caps": {"wall_s": BURST_MAX_S, "steps": BURST_MAX_STEPS},
             "cooldown_s": COOLDOWN_S,
             "cpu_threads": torch.get_num_threads(),
             "polls_logged": "run log + runs/_envelope_log.jsonl (every "
                             "poll)",
             "n_launch_poll_cycles": len(envelope_poll_log) // 2,
             "wash_executed_this_pass": bool(todo),
             "cpu_load_checks": load_checks,
             "pass": True}
    if not todo:
        # the plot-only re-pass (the e182c2 precedent): the wash itself ran
        # in the prior pass — recover its envelope record from the ON-DISK
        # audit trail (the run log's poll/burst lines; runs/
        # _envelope_log.jsonl holds the same polls), recording-only
        G_ENV["wash_pass_note"] = (
            "the wash ran in the prior DONE pass (455.2s wall; see "
            "runs/e217_run.log + runs/_envelope_log.jsonl); THIS pass is "
            "the plot-only re-pass off the frozen journal — zero GPU, zero "
            "wash recompute, the census numbers bit-identical")
        try:
            logtxt = LOG_PATH.read_text(encoding="utf-8", errors="replace")
            n_polls = logtxt.count("[gpu:burst")
            bursts = [ln.split("] ", 1)[1] for ln in logtxt.splitlines()
                      if "burst done:" in ln]
            G_ENV["carried_from_wash_pass"] = {
                "n_launch_poll_cycles": n_polls // 2,
                "n_polls_logged": n_polls,
                "burst_walls": bursts,
                "source": f"parsed from {LOG_PATH} (the append-mode audit "
                          "trail; re-passes add no GPU polls)",
            }
        except Exception as e:                          # noqa: BLE001
            G_ENV["carried_from_wash_pass"] = {"error": repr(e)}
    metrics["gates"]["G_ENV"] = G_ENV
    write_metrics("PARTIAL: the third draw complete; the census pending")

    if SMOKE:
        metrics["adjudication"] = {
            "bars": REGISTERED_PREDICTION["bars_verbatim"],
            "fired": [], "verdict_composite": "SMOKE (nothing adjudicated)",
            "clause": "smoke run"}
        write_metrics("SMOKE DONE (census + adjudication skipped)")
        log("SMOKE DONE — nothing adjudicated")
        return 0

    # --------------------------------------------- P5 the three-wash census
    st3 = {s["step"]: s for s in p3_states}
    retention["w3"] = {s["step"]: {b: s[b]["mean_p"] for b in BATTERIES}
                       for s in p3_states}
    decl3 = {str(s["step"]): {b: 1 - s[b]["mean_p"] / base_f[b]
                              for b in BATTERIES}
             for s in p3_states if s["step"] > 0}
    census = {
        "note": "declines w1/w2 RUNTIME-READ from runs/e213/metrics.json "
                "(census.declines); w3 computed here (CPU fp32 probes)",
        "states": list(CENSUS_STATES),
        "declines": {"w1": decl_rec["w1"], "w2": decl_rec["w2"],
                     "w3": decl3},
        "pairwise": {},
        "floor_cells_plus10": {},
    }
    for s_ in CENSUS_STATES:
        key = str(s_)
        cell = {}
        for b in BATTERIES:
            d1 = decl_rec["w1"][key][b]
            d2 = decl_rec["w2"][key][b]
            d3 = decl3[key][b]
            r21 = d2 / d1 if d1 > DECL_FLOOR else None
            r31 = d3 / d1 if d1 > DECL_FLOOR else None
            r32 = d3 / d2 if d2 > DECL_FLOOR else None
            cell[b] = {"decl_w1": d1, "decl_w2": d2, "decl_w3": d3,
                       "r21": r21, "r31": r31, "r32": r32,
                       "denominator_floor": bool(d1 <= DECL_FLOOR)}
        census["pairwise"][key] = cell
        # the +10 shallow row co-report (e213's floor rule, w3 arm)
        fc = {}
        for b in BATTERIES:
            d1 = decl_rec["w1"][key][b]
            d3 = decl3[key][b]
            if d1 <= DECL_FLOOR:
                fc[b] = {"floor_cell": True,
                         "abs_diff_w3w1": abs(d3 - d1),
                         "verdict": "FLOOR-MATCH" if abs(d3 - d1) <= 0.02
                                    else "FLOOR-WANDER"}
            else:
                fc[b] = {"floor_cell": False}
        census["floor_cells_plus10"][key] = fc
    metrics["census"] = census

    # ------------------------------------------ P6 the adjudication (frozen)
    gates_ok = bool(G_SIZE["pass"] and G_CORPUS["pass"] and G_BATT["pass"]
                    and G_TMPL["pass"] and G_STATES_PROV["pass"]
                    and G_PPL["pass"])
    adj = {"bars": REGISTERED_PREDICTION["bars_verbatim"],
           "gates_summary": {"G_SIZE": G_SIZE["pass"],
                             "G_CORPUS": G_CORPUS["pass"],
                             "G_BATT": G_BATT["pass"],
                             "G_TMPL": G_TMPL["pass"],
                             "G_STATES_PROV": G_STATES_PROV["pass"],
                             "G_PPL": G_PPL["pass"],
                             "G_ENV": G_ENV["pass"]}}
    if not gates_ok:
        failed = [g for g, v in adj["gates_summary"].items() if not v]
        adj.update({"fired": [],
                    "verdict_composite": "VERIFICATION-FAILED (curves "
                                         "reported; no bar read)",
                    "mid_dose": {"fires": None},
                    "deep_drift": {"deepest_batteries": None},
                    "clause": "verification gates failed: "
                              + ", ".join(failed)})
        metrics["adjudication"] = adj
        write_metrics("DONE (verification failed; no bar read)")
        log("VERIFICATION-FAILED — no bar read")
        return 0

    # ---- MID-DOSE-TIGHTENS (at +50)
    mid_cells = census["pairwise"][str(MID_STATE)]
    denom_ok = not any(any(mid_cells[b][r] is None
                           for r in ("r21", "r31", "r32"))
                       for b in BATTERIES)
    devs = {b: max(abs(mid_cells[b][r] - 1)
                   for r in ("r21", "r31", "r32"))
            for b in BATTERIES} if denom_ok else {}
    devs_wo3 = {b: abs(mid_cells[b]["r21"] - 1) for b in BATTERIES}
    max_dev = max(devs.values()) if devs else None
    mid_fires = bool(denom_ok and max_dev is not None
                     and max_dev <= RATIO_TOL_MID)
    adj["mid_dose"] = {
        "state": MID_STATE,
        "fires": mid_fires,
        "max_abs_dev_all_pairs": max_dev,
        "tol": RATIO_TOL_MID,
        "per_battery_max_abs_dev": devs,
        "per_battery_abs_dev_w2_only": devs_wo3,
        "spread_shrinks_with_third_draw":
            bool(devs and max_dev <= max(devs_wo3.values())),
        "denominator_guard_ok": denom_ok,
        "note": "fires iff max |ratio-1| over the three pairwise ratios "
                "{r21, r31, r32} <= 0.15 for EVERY battery; the w2-only "
                "co-reading (e213's r21) makes 'shrinks' readable",
    }

    # ---- the DEEP bars (at +80)
    deep_cells = census["pairwise"][str(DEEP_STATE)]
    deepest = sorted(BATTERIES,
                     key=lambda b: -decl_rec["w1"][str(DEEP_STATE)][b])[:2]
    deep = {"state": DEEP_STATE, "deepest_batteries": deepest,
            "deepest_w1_declines": {b: decl_rec["w1"][str(DEEP_STATE)][b]
                                    for b in deepest},
            "per_battery": {}, "fired": None, "clause": ""}
    for b in deepest:
        r21 = deep_cells[b]["r21"]
        r31 = deep_cells[b]["r31"]
        d2, d3 = abs(r21 - 1), abs(r31 - 1)
        same_sign = (r21 - 1) * (r31 - 1) > 0
        grows = bool(same_sign and d3 > d2)
        reverts = bool((not same_sign) or d3 < d2)
        deep["per_battery"][b] = {
            "r21": r21, "r31": r31, "dist2": d2, "dist3": d3,
            "same_sign": same_sign, "GROWS": grows, "REVERTS": reverts,
            "FLAT": bool(not grows and not reverts),
        }
    grows_all = all(deep["per_battery"][b]["GROWS"] for b in deepest)
    reverts_all = all(deep["per_battery"][b]["REVERTS"] for b in deepest)
    if grows_all:
        deep["fired"] = "DEEP-DRIFT-GROWS"
    elif reverts_all:
        deep["fired"] = "DEEP-MEAN-REVERTS"
    else:
        deep["fired"] = "GRADED"
    adj["deep_drift"] = deep

    # ---- the composite
    fired = []
    if mid_fires:
        fired.append("MID-DOSE-TIGHTENS")
    if deep["fired"] in ("DEEP-DRIFT-GROWS", "DEEP-MEAN-REVERTS"):
        fired.append(deep["fired"])
    graded = bool(deep["fired"] == "GRADED" or not denom_ok
                  or not mid_fires)
    if graded:
        fired.append("GRADED")
    adj["fired"] = fired
    adj["verdict_composite"] = " + ".join(f for f in fired if f != "GRADED") \
        + (" + GRADED (partial — the tables verbatim)" if graded else "")
    if not fired:
        adj["verdict_composite"] = "no bar fired (mid failed cleanly; the " \
                                   "tables verbatim)"

    mrow = " ".join(f"{b} {devs[b]:.3f}" for b in BATTERIES) if devs \
        else "denominator guard tripped (a +50 decline at the floor)"
    deep_bits = "; ".join(
        f"{b}: r21 {deep['per_battery'][b]['r21']:.3f} -> r31 "
        f"{deep['per_battery'][b]['r31']:.3f} "
        f"({'GROWS' if deep['per_battery'][b]['GROWS'] else ('REVERTS' if deep['per_battery'][b]['REVERTS'] else 'FLAT')})"
        for b in deepest)
    max_dev_txt = f"{max_dev:.3f}" if max_dev is not None else "n/a"
    w2only_txt = f"{max(devs_wo3.values()):.3f}" if devs_wo3 else "n/a"
    adj["clause"] = (
        f"MID-DOSE at +{MID_STATE}: max|r-1| over the three pairwise "
        f"ratios = {max_dev_txt} (tol {RATIO_TOL_MID}; per battery {mrow}; "
        f"e213's w2-only max {w2only_txt}) -> "
        f"{'FIRES (the mid-dose state function licensed at n=3)' if mid_fires else 'does NOT fire'}. "
        f"DEEP at +{DEEP_STATE}: the deepest batteries {deepest} "
        f"(wash-1 declines "
        f"{', '.join(f'{b} {deep['deepest_w1_declines'][b]:.3f}' for b in deepest)}) "
        f"{deep_bits} -> {deep['fired']}.")
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E217 VERDICT: {adj['verdict_composite']}")
    log(f"  {adj['clause']}")
    write_metrics("PARTIAL: the census adjudicated; the relational n=3 "
                  "pending")

    # ----------------------------- P7 the relational signature at n=3 (co-report)
    s80 = st3[80]
    per_probe = []
    for rec in e215m["typology"]["assignment"]:
        f = rec["fact"]
        fam = rec["family6"]
        p0 = p3_states[0][rec["battery"]]["probes"].get(f, {}).get("p")
        p80 = s80[rec["battery"]]["probes"][f]["p"] \
            if f in s80[rec["battery"]]["probes"] else None
        assert p0 is not None and p80 is not None, f"probe {f} missing"
        per_probe.append({
            "fact": f, "battery": rec["battery"], "family": fam,
            "p0": p0, "p80_w3": p80, "hr_w3": p80 / p0,
            "hr_w1": rec["hr_w1"], "hr_w2": rec["hr_w2"],
            "hr_w1_s50": rec.get("hr_w1_s50"), "hr_w2_s50": rec.get("hr_w2_s50"),
            "hr_w3_s50": (st3[50][rec["battery"]]["probes"][f]["p"] / p0
                          if f in st3[50][rec["battery"]]["probes"] else None),
            "p0_e215": rec["p0"],
        })
    fams = sorted({r["family"] for r in per_probe})
    fam_meds = {f: {"w1": median([r["hr_w1"] for r in per_probe
                                  if r["family"] == f]),
                    "w2": median([r["hr_w2"] for r in per_probe
                                  if r["family"] == f]),
                    "w3": median([r["hr_w3"] for r in per_probe
                                  if r["family"] == f])} for f in fams}
    fam_meds50 = {f: {"w1": median([r["hr_w1_s50"] for r in per_probe
                                    if r["family"] == f]),
                      "w2": median([r["hr_w2_s50"] for r in per_probe
                                    if r["family"] == f]),
                      "w3": median([r["hr_w3_s50"] for r in per_probe
                                    if r["family"] == f])} for f in fams}
    fam_spread = {f: {"n": sum(1 for r in per_probe if r["family"] == f),
                      "medians": fam_meds[f],
                      "spread_max_min": max(fam_meds[f].values())
                      - min(fam_meds[f].values())} for f in fams}
    fam_class3 = {f: hold_class(median([r["hr_w3"] for r in per_probe
                                        if r["family"] == f])) for f in fams}
    cls3 = {}
    for r in per_probe:
        k = "/".join(hold_class(r[f"hr_w{w}"]) for w in (1, 2, 3))
        cls3[k] = cls3.get(k, 0) + 1
    class3_counts = {
        "HOLD on all 3": sum(1 for r in per_probe
                             if all(hold_class(r[f"hr_w{w}"]) == "HOLD"
                                    for w in (1, 2, 3))),
        "COLLAPSE on all 3": sum(1 for r in per_probe
                                 if all(hold_class(r[f"hr_w{w}"]) == "COLLAPSE"
                                        for w in (1, 2, 3))),
        "discordant (any)": sum(1 for r in per_probe
                                if len({hold_class(r[f"hr_w{w}"])
                                        for w in (1, 2, 3)}) > 1),
        "n_total": len(per_probe),
    }
    rel = {
        "note": "CO-REPORT (no bar): the relational signature's cross-wash "
                "reliability at n=3 — hr_w = p_w(+80)/p_0; the 6-family "
                "typology + hr_w1/hr_w2 RUNTIME-READ from runs/e215/"
                "metrics.json; hr_w3 from THIS wash; family statistic = "
                "MEDIAN hr (e215's instrument)",
        "family_order": fams,
        "family_medians": fam_meds,
        "family_medians_s50_co_report": fam_meds50,
        "family_spread": fam_spread,
        "family_class3_w3": fam_class3,
        "spearman": {
            "w2xw1": spearman([r["hr_w2"] for r in per_probe],
                              [r["hr_w1"] for r in per_probe]),
            "w3xw1": spearman([r["hr_w3"] for r in per_probe],
                              [r["hr_w1"] for r in per_probe]),
            "w3xw2": spearman([r["hr_w3"] for r in per_probe],
                              [r["hr_w2"] for r in per_probe]),
            "note": "pooled per-probe; w2xw1 reproduces e215's 0.939 "
                    "(its instrument); n=54",
        },
        "class3_counts": class3_counts,
        "class3_confusion": cls3,
        "per_probe": per_probe,
    }
    metrics["relational_n3"] = rel
    log("relational n=3: family medians (w1/w2/w3): "
        + "; ".join(f"{f} {fam_meds[f]['w1']:.2f}/{fam_meds[f]['w2']:.2f}/"
                    f"{fam_meds[f]['w3']:.2f}" for f in fams)
        + f"; rho w3xw1 {rel['spearman']['w3xw1']:.3f}, w3xw2 "
        f"{rel['spearman']['w3xw2']:.3f}")

    # ------------------------------------------------------------ the honesty
    metrics["honesty_reflex"] = {
        "n_counts": f"n=3 wash draws (18202 CPU-replay arm, 20261002 GPU "
                    f"fp32, {FRESH_SEED} GPU fp32); fact n=20, ctrl n=12, "
                    "near n=3 (item-level noise the 19-20 item batteries "
                    "do not carry), tmpl n=19; single organism (gpt2 124M)",
        "device_texture": "wash 1's states are e182c's CPU fp32 REPLAY of "
            "e182's GPU wash (G_REPLAY stamp <= 0.001 fact mean_p dev at "
            "shared steps); washes 2 and 3 trained GPU fp32 (TF32 OFF) — "
            "the census compares declines (patterns), never bit values",
        "deepest_frozen": f"the deepest batteries {deepest} are FROZEN from "
            "the committed e213 record (the two largest wash-1 +80 "
            "declines) — never re-derived from wash 3",
        "n3_is_still_texture": "n=3 draws separates grow from revert no "
            "better than one more coin flip weights it; a tightening "
            "mid-dose does not prove state-functionality (three draws can "
            "coincide on a loose band); the openness is the point",
        "shallow_region": "the +10 row is co-reported only (denominator-"
            "fragile; e213's floor rule applied to the w3 arm); the "
            "registered bars live at +50 and +80",
        "relational_instrument": "the family label is e215's hand-registered "
            "typology (an instrument, not a finding); hr divides by p0 as "
            "low as ~0.52 — deep-state floors compress ratios",
        "path_scope": "'path' = the training-data-order path at fixed dose "
            "(lr/corpus/recipe/organism held); the census tests "
            "order-sensitivity of the dose-response, not path-"
            "independence in the thermodynamic sense",
    }
    metrics["compute"] = {
        "wash3": f"GPU fp32 fresh wash (seed {FRESH_SEED}), {FRESH_STEPS} "
                 f"steps in owner-envelope bursts (<= {BURST_MAX_STEPS} "
                 f"steps / <= {BURST_MAX_S:.0f}s), cooldown >= "
                 f"{COOLDOWN_S:.0f}s, double-polled launches; probes CPU "
                 f"fp32 threads {torch.get_num_threads()}",
        "state_archive": [f"runs/checkpoints/{NAME}_fresh_s{s}.pt"
                          for s in FRESH_CK],
        "resumable": f"runs/checkpoints/{NAME}_fresh_latest.pt",
        "records_runtime_read": ["runs/e213/metrics.json (w1/w2 declines "
                                 "+ census states + t=0 anchor + the "
                                 "archive inventory)",
                                 "runs/e215/metrics.json (the family "
                                 "typology + hr_w1/hr_w2)",
                                 "runs/e182/metrics.json (the corpus record)"],
        "run_log": str(LOG_PATH),
        "resumable_journal": str(jp),
    }
    metrics["trims"] = []
    metrics["deviations"] = deviations

    # ---------------------------------------------------------------- the plots
    # (drawn BEFORE the DONE stamp; a plot failure is recorded, never a
    # silent crash — the e182c2 plot-only-re-pass precedent, pre-empted)
    pngs = []
    plot_status = "DONE"
    retention_arg = {w: {s: retention[w][s] for s in sorted(retention[w])}
                     for w in ("w1", "w2", "w3")}
    for label, fn in (("three_wash_census",
                       lambda: plot_three_wash(rd, census, adj,
                                               retention_arg,
                                               MID_STATE, DEEP_STATE)),
                      ("relational_signature_n3",
                       lambda: plot_relational(rd, rel))):
        try:
            pngs.append(fn())
        except Exception as e:                          # noqa: BLE001
            plot_status = (f"DONE (plot {label} FAILED: {e!r} — metrics "
                           "complete; re-run is plot-only)")
            deviations.append(f"plot failure ({label}): {e!r} — recorded, "
                              "not fatal (the numbers are frozen in "
                              "metrics.json + the journal)")
            log(f"PLOT FAILURE ({label}): {e!r}")
    metrics["plot_outputs"] = [str(p) for p in pngs]
    write_metrics(plot_status)
    log(f"outputs: {rd / 'metrics.json'}, {pngs}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
