"""E182C PHASE 2 — THE TEMPLATE LOCUS + THE FRESH DRAWS (e182c's unlock).

WHY: e182c phase 1 (runs/e182c; T149) retired the surgical signature at
124M — FORGETTING-GENERIC fired: matched controls erode WITH the facts
(ratio 0.92 at +80) while wash perplexity improves. What phase 1 OPENED is
the template-locus hint: the near-related battery (the fact template "The
capital of {c} is" over DISJOINT entities, US states, n=3) collapsed
FASTEST of all (retention 0.234 at +80, decline 0.766 vs facts 0.429 /
controls 0.397). Erosion may live at the few-shot-template-following
level rather than knowledge storage — but with ONE form probed, "the
capital-of form" and "the few-shot-following faculty" are confounded.
Phase 2 breaks that with two moves on the states phase 1 saved:

THE CELL:
  (1) THE TEMPLATE CONTROL at the deeper state — a SECOND matched battery
      with a DIFFERENT template form (the reversed copular: "The state
      whose capital is {a} is" -> the STATE; the dispatch's own example
      form, applied over the held-out US-state family so the battery
      stays disjoint from the fact battery exactly as nearrel did),
      evaluated EVAL-ONLY on phase 1's SAVED wash states
      (runs/checkpoints/e182c_s{2,10,50,80}.pt — the discipline phase 1
      left us). Does the template-level erosion generalize across
      template FORMS or is it capital-of-form-specific?
  (2) THE FRESH CORPUS DRAW — ONE fresh wash draw at 5e-5 (the SAME
      frozen corpus + recipe; the ONLY delta is a NEW registered stream
      seed 20261002), ~80 steps in GPU short bursts, reading the fact
      battery + phase-1's control battery + the new template battery at
      t=0/+10/+50 (+80 co-reported, phase 1's primary depth). Does the
      phase-1 pattern (generic erosion; the near-related fastest)
      replicate on a fresh draw, or was that texture one draw's?

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any wash
compute; the screening below used t=0 pristine information ONLY; no bar
shopping — adjudicate against exactly this):
  - TEMPLATE-GENERAL: "fires if the second template's controls erode
    comparably to the first's (within ~1.5x at the deepest shared state)
    — the erosion is template-GENERAL; the locus is the few-shot-following
    faculty, not one form."
  - TEMPLATE-SPECIFIC: "fires if the second template's controls hold
    materially better — the erosion is template-specific; the hint
    rescopes to the capital-of form."
  - DRAW-REPLICATES: "fires if the fresh draw reproduces the phase-1
    pattern (generic erosion ratio in [0.7, 1.3]; the near-related still
    fastest) — the texture at n=2 draws."
  - DRAW-DIFFERS: "fires if the fresh draw's pattern differs materially —
    the phase-1 texture was one draw's; reported honestly."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * batteries: "the first" template battery := phase 1's NEARREL battery
    (the capital-of form over US states, n=3, declines read at RUNTIME
    from runs/e182c/metrics.json — never transcribed); "the second" :=
    THE TEMPLATE BATTERY here (reversed form over US states, the frozen
    30-pair pool below, e182's gate VERBATIM: (top-1 and p>=0.8) or
    (top-5 and p>=0.5), cap 20 by baseline p; n=whatever passes, floor 6
    flagged reduced — nearrel itself ran at n=3 as a co-report, so no
    abort).
  * decline_b(s) = 1 - R_b(s)/R_b(0), R = battery mean p(answer first
    token) — e182's recall ruler, one absolute ruler per battery,
    retention normalized per battery.
  * "the deepest shared state" = +80 (phase 1's own deepest probed state;
    both batteries read there; the +50 co-adjudication is co-reported).
  * "within ~1.5x" := near80/1.5 <= tmpl80 <= near80*1.5 AND near80 >
    0.02 (the first battery must actually erode). "Hold materially
    better" := tmpl80 < near80/1.5. If tmpl80 > near80*1.5 (erodes even
    faster than the band): neither template bar -> GRADED, tables
    verbatim.
  * DRAW (at +50, the registered fresh grid's deepest; +80 co-adjudicated
    alongside): ratio := ctrl_decl/fact_decl on the fresh draw, requiring
    fact_decl > 0.02 (else clause A FAILS, disclosed — a non-eroding
    fresh draw IS a material difference); clause A := 0.7 <= ratio <=
    1.3; clause B := nearrel_decl > fact_decl AND nearrel_decl >
    ctrl_decl (phase 1's cast, like-for-like; the template battery's
    decline co-reported, and a secondary note fires if the template
    battery outpaces nearrel). REPLICATES := A AND B; DIFFERS := (not A)
    AND (not B); exactly one of A/B -> GRADED (partial; tables verbatim).
  * Adjudication is GATED on the verification gates below; if any fails:
    VERIFICATION-FAILED, curves reported, no bar read.

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_STATES — phase 1's saved states exist (runs/checkpoints/
    e182c_s{2,10,50,80}.pt; sizes/mtimes recorded) and RE-VERIFY: ctrl +
    nearrel re-probed on the LOADED s50/s80 must match phase 1's recorded
    per-probe p within 0.005 (expected ~0.0 — same fp32 weights, same CPU
    fp32 probe path). This is the provenance gate for the whole of (1).
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to e182's
    recorded filter stats AND phase 1's corpus record (40001 lines / 664
    dropped / 1093972 chars / 331770 tokens / train 319481 / bank 24x512,
    banned list identical). The corpus is INHERITED FROZEN — the fresh
    draw changes ONLY the stream seed.
  * G_BATT — the fact/control/nearrel batteries rebuilt VERBATIM
    (phase-1 module import) must reproduce phase 1's t=0 readings: same
    kept sets, per-probe |dp| <= 0.010, battery-mean |dR0| <= 0.005.
  * G_TMPL — the template battery's gate discipline (e182 VERBATIM),
    contamination scans zero for every KEPT item (state and capital
    strings absent from the frozen filtered corpus; answer token id
    absent from the exact training stream), zero overlap with e182's 63
    banned fact strings, single-token answers, floor-6 flag.
  * G_PPL — the held-out bank perplexity read at every fresh state and
    IMPROVES (the health reference phase 1 ran).
  * G_ENV — the owner envelope: every GPU launch double-polled
    (util <= 20% AND temp <= 70C, >=5 s apart), every poll logged to the
    run log AND runs/_envelope_log.jsonl; bursts <= 75 s wall; cooldown
    >= 180 s between bursts; no GPU outside bursts.

SCREENING PROVENANCE (t=0 pristine-model information ONLY, AFTER the
bars were frozen and BEFORE any wash compute; the phase-1 precedent):
the template battery's pool was screened in ONE round with the frozen
draft order below. Result (pristine t=0, fixed exemplars pool[0]+pool[1]
= Boston->Massachusetts 0.886 / Atlanta->Georgia 0.963): 7 of 30 dropped
pre-probe by the contamination scan (capitals Phoenix/Salem/Olympia/
Montgomery/Saint Paul/Helena in the frozen corpus; state 'Maine' in the
corpus), 0 multi-token (all 40 single-word US states are single GPT-2
tokens — verified), 23 probed, 19 PASS (p 0.693-0.988, mean 0.774),
n=19 kept (cap 20 not binding; above the floor-10 unflagged line —
LARGER than phase 1's n=3 nearrel and n=12 controls). The reversed
direction is NOT scarce at 124M — itself a datum (state-identity cloze
is at least as strong as the forward form). Pool order FROZEN as
screened; never tuned again.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the openness is
the point. The reversed form changes BOTH surface form AND direction
(state->capital vs capital->state): a GENERAL fire means form-robustness
of the fast erosion, not that "the faculty" is the locus (direction is
confounded with form in this one battery); a SPECIFIC fire rescopes the
fast collapse to the capital-of FORM but over a first battery of n=3.
The fresh draw is n=1 additional draw (total n=2 washes): texture, not
law. HONESTY STAMPS: part (1) reads phase-1's CPU fp32 replay states
(steps 1-25 of that replay deviated from e182's GPU original by <= 0.001
in fact mean_p — G_REPLAY phase 1); part (2)'s wash trains on GPU fp32
(TF32 OFF, matmul precision 'highest') where phase 1's replay was CPU
fp32 — a device-texture deviation, disclosed; probes are CPU fp32
everywhere (phase-1's instrument, bit-identical path); the template
battery n=1 pool, nearrel n=3; 124M inference cost: ~54 prompts + 24x512
bank per state ≈ 60-120 s CPU per state, walls logged.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02, permanent): the lab is
the LOWEST priority — GPU launch only on double-polled util <= 20% /
temp <= 70C; bursts <= 75 s (the 80-step wash split into <=40-step
bursts); cooldown >= 180 s between bursts (CPU probing between bursts
counts toward it); every poll logged; CPU threads capped at 8 (e182
convention; e211 shares the CPU). No NOTES/THINKING/QUEUE/STATE edits
(dispatch). Progressive PARTIAL metrics + resumable journals after every
state (the standing disruption rule).

PROVENANCE: the organism, the frozen corpus filter+verify, the wash
recipe (AdamW (0.9,0.95) wd 0.1 constant lr 5e-5, clip 1.0, batch 8 x
ctx 512, CPU-generator window draws), the batteries (fact/control/
nearrel), probe_battery/select_battery/ppl_eval are lab/
e182c_forgetting_control.py VERBATIM via module import (which itself
inherits lab/e182_gpt2_wash.py). Phase-1's recorded curves are read at
runtime from runs/e182c/metrics.json (never transcribed). Builds on:
e182c phase 1 / T149 (the unlock + the saved states), e182/T123 (the
parent wash), SUPERVISOR check-in 12 item 3 (the control letter). NEW:
the reversed-form template battery, the saved-state eval-only re-read,
the fresh stream seed (20261002 — registered here before compute), the
template-locus and fresh-draw adjudications.

Run:  cd lab && python e182c2_template.py   (E182C2_SMOKE=1: 2-step
      fresh wash, grid {1,2}, own smoke dir, nothing adjudicated)
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
torch.set_num_threads(8)                              # e182 convention

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import now_iso, run_dir, save_json          # noqa: E402

import e182c_forgetting_control as e1                  # noqa: E402 — the phase-1 machinery, VERBATIM

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E182C2_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e182c2_smoke" if SMOKE else "e182c2"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen phase-2 constants (registered BEFORE compute) --------------
FRESH_SEED = 20261002          # the NEW stream seed (e182/phase-1 was 18202)
FRESH_STEPS = 2 if SMOKE else 80
FRESH_CK: tuple[int, ...] = (1, 2) if SMOKE else (10, 50, 80)
DRAW_STEP = 2 if SMOKE else 50          # the registered DRAW adjudication depth
DRAW_STEP_DEEP = 2 if SMOKE else 80     # the co-adjudication depth
SAVED_STATES: tuple[int, ...] = (1, 2) if SMOKE else (2, 10, 50, 80)

# the owner envelope (stricter than common's gpu_ok launch gate)
LAUNCH_UTIL, LAUNCH_TEMP = 20.0, 70.0
POLL_GAP_S = 5.0
BURST_MAX_S = 75.0
BURST_MAX_STEPS = 40
COOLDOWN_S = 180.0

# ---- the template battery (THE new instrument; pool frozen as screened) ----
# The reversed copular form of the capital-of cloze: the prompt names the
# CAPITAL, the answer is the STATE. US states keep the battery disjoint from
# the fact battery's country pools exactly as phase-1's nearrel did.
TMPL2 = ("The state whose capital is {a} is {c}. ",
         "The state whose capital is {a} is")
TMPL_POOL: list[tuple[str, str]] = [   # (state, capital); FROZEN as screened
    ("Massachusetts", "Boston"), ("Georgia", "Atlanta"),
    ("California", "Sacramento"), ("Texas", "Austin"), ("Ohio", "Columbus"),
    ("Colorado", "Denver"), ("Arizona", "Phoenix"), ("Illinois", "Springfield"),
    ("Oregon", "Salem"), ("Washington", "Olympia"),
    ("Alabama", "Montgomery"), ("Alaska", "Juneau"),
    ("Arkansas", "Little Rock"), ("Connecticut", "Hartford"),
    ("Delaware", "Dover"), ("Florida", "Tallahassee"),
    ("Hawaii", "Honolulu"), ("Idaho", "Boise"), ("Indiana", "Indianapolis"),
    ("Iowa", "Des Moines"), ("Kansas", "Topeka"), ("Kentucky", "Frankfort"),
    ("Louisiana", "Baton Rouge"), ("Maine", "Augusta"),
    ("Maryland", "Annapolis"), ("Michigan", "Lansing"),
    ("Minnesota", "Saint Paul"), ("Mississippi", "Jackson"),
    ("Missouri", "Jefferson City"), ("Montana", "Helena"),
]
SCREENING_RECORD = {
    "rounds": 1, "when": "after bars frozen, before any wash compute",
    "information": "t=0 pristine model ONLY (phase-1 precedent)",
    "dropped_pre_probe": [
        "Arizona/Phoenix — capital 'Phoenix' in the frozen corpus",
        "Oregon/Salem — capital 'Salem' in the frozen corpus",
        "Washington/Olympia — capital 'Olympia' in the frozen corpus",
        "Alabama/Montgomery — capital 'Montgomery' in the frozen corpus",
        "Maine/Augusta — state 'Maine' in the frozen corpus",
        "Minnesota/Saint Paul — capital 'Saint Paul' in the frozen corpus",
        "Montana/Helena — capital 'Helena' in the frozen corpus",
    ],
    "multi_token_dropped": 0,
    "n_probed": 23, "n_passed": 19, "n_kept_expected": 19,
    "p0_range": "0.693-0.988", "p0_mean": 0.774,
    "fixed_exemplars": "pool[0]+pool[1] = Boston->Massachusetts (0.886) + "
                       "Atlanta->Georgia (0.963) — both strong passers",
    "note": "the reversed direction is NOT scarce at 124M — state-identity "
            "cloze is at least as strong as the forward form; pool order "
            "frozen as screened, never tuned again",
}

# ---- registered bar constants (frozen) --------------------------------------
TEMPLATE_BAND = 1.5            # "within ~1.5x" (phase-1's GEN_BAND)
DECL_FLOOR = 0.02              # the first battery / the facts must erode
DRAW_RATIO_LO, DRAW_RATIO_HI = 0.7, 1.3

# ---- registered verification tolerances (frozen) -----------------------------
TOL_PROBE_DP = 0.010           # per-probe t=0 dp vs phase-1 record
TOL_R0_DP = 0.005              # battery-mean t=0 dp
TOL_STATE_DP = 0.005           # G_STATES re-probe dp on loaded s50/s80

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "TEMPLATE-GENERAL": "fires if the second template's controls erode "
            "comparably to the first's (within ~1.5x at the deepest shared "
            "state) — the erosion is template-GENERAL; the locus is the "
            "few-shot-following faculty, not one form.",
        "TEMPLATE-SPECIFIC": "fires if the second template's controls hold "
            "materially better — the erosion is template-specific; the hint "
            "rescopes to the capital-of form.",
        "DRAW-REPLICATES": "fires if the fresh draw reproduces the phase-1 "
            "pattern (generic erosion ratio in [0.7, 1.3]; the near-related "
            "still fastest) — the texture at n=2 draws.",
        "DRAW-DIFFERS": "fires if the fresh draw's pattern differs "
            "materially — the phase-1 texture was one draw's; reported "
            "honestly.",
        "GRADED": "any partial — the tables verbatim.",
    },
    "lean": "phase-1's hint leans TEMPLATE-GENERAL (the few-shot-following "
        "faculty) and DRAW-REPLICATES (the +50 ratio 0.83 sat mid-band and "
        "the nearrel lead was stark); but the first battery is n=3, the "
        "reversed form confounds form with direction, and a fresh draw is "
        "n=1 — nothing guaranteed; the openness is the point",
    "operationalizations": "first := phase-1 nearrel (capital-of form, n=3, "
        "declines runtime-read); second := the reversed-form template "
        "battery (frozen 30-pair pool, e182 gate verbatim, cap 20, floor 6 "
        "flagged); decline_b(s)=1-R_b(s)/R_b(0); deepest shared state +80 "
        "(+50 co-reported); within-1.5x := near80/1.5 <= tmpl80 <= "
        "near80*1.5 AND near80>0.02; hold-materially-better := tmpl80 < "
        "near80/1.5; tmpl80 > near80*1.5 -> GRADED; DRAW at +50 (+80 "
        "co-adjudicated): A := 0.7 <= ctrl_decl/fact_decl <= 1.3 (needs "
        "fact_decl>0.02, else A fails disclosed); B := nearrel_decl > "
        "fact_decl AND > ctrl_decl (phase-1 cast); REPLICATES := A AND B; "
        "DIFFERS := (not A) AND (not B); exactly one -> GRADED; adjudication "
        "gated on G_STATES/G_CORPUS/G_BATT/G_TMPL/G_PPL",
    "registration": "bars frozen VERBATIM from the dispatch brief; the fresh "
        "stream seed 20261002 and the template pool registered in this file "
        "and committed BEFORE any wash compute; no bar shopping",
}

deviations: list[str] = [
    "Part (1) reads phase-1's SAVED states (the discipline phase 1 "
    "introduced): eval-only, no wash re-run, admitted by the G_STATES "
    "re-probe verification (expected bit-identical: same fp32 weights, same "
    "CPU fp32 probe path).",
    "The template battery's reversed form changes BOTH surface form AND "
    "direction (capital->state vs state->capital) — the dispatch's own "
    "example form; a GENERAL fire is form-robustness of the erosion, not "
    "proof the 'faculty' (vs the direction) is the locus. Disclosed before "
    "compute.",
    "Part (2)'s wash trains on GPU fp32 (TF32 OFF; matmul precision "
    "'highest'); phase-1's wash was a CPU fp32 replay. The draw discipline "
    "(CPU-generator window offsets) is bit-identical; optimizer arithmetic "
    "texture differs by device — disclosed, and part of why the DRAW bars "
    "compare PATTERNS (declines/ratios), never bit values.",
    "Fresh-draw per-state weights saved (runs/checkpoints/"
    "e182c2_fresh_s{N}.pt + resumable _latest.pt) so no future phase "
    "re-runs this wash.",
    "The nearrel battery (n=3) is re-read on the fresh draw as the "
    "like-for-like instrument for the DRAW-B clause ('the near-related "
    "still fastest' — phase-1's cast); the template battery's fresh-draw "
    "decline is co-reported with a secondary note if it outpaces nearrel.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: 2-step fresh wash, grids {1,2}, own smoke dir and log, "
    "nothing adjudicated or verified.",
    "PLOT-ONLY RE-PASS (the phase-1 precedent): the first full pass wrote "
    "DONE metrics + journals and then crashed in the SECOND figure "
    "(adj2['ratio'] KeyError — the ratio lived in the draw_bars record, "
    "not at top level). The fix is plot-only + recording-only (the "
    "draw_bars booleans added to the adjudication dict); main() re-ran "
    "SELF-RESUMING off the frozen journals — zero wash recompute, zero "
    "GPU, probes deterministic CPU fp32 — so every number in metrics.json "
    "is bit-identical to the crashed pass's; the envelope polls of the "
    "actual wash live in runs/e182c2_run.log + runs/_envelope_log.jsonl "
    "(3 launch cycles, 6 polls, all FREE).",
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


# ------------------------------------------------------- the template battery

def build_tmpl_candidates(tok, filtered_lower: str, train_ids,
                          e182_banned: list[str]):
    """The REVERSED-form battery (near-verbatim adaptation of e1's
    build_control_candidates for a template whose query fills {a}, not
    {c}): drops in order — (1) overlap with e182's banned strings, (2)
    corpus contamination (state or capital string in the FROZEN filtered
    corpus, case-insensitive; answer token id in the exact training
    stream), (3) multi-token answers (e182's rule). Survivors probed with
    e182's fixed-exemplar mechanics (pool[0]+pool[1] beyond the first two
    probes), then the VERBATIM gate."""
    probed, dropped = [], []
    banned_set = set(e182_banned)
    pool = TMPL_POOL
    sent, query = TMPL2
    for i, (state, cap) in enumerate(pool):
        reason = None
        sl, cl = state.lower(), cap.lower()
        if sl in banned_set or cl in banned_set:
            reason = (f"overlap with e182's banned fact strings "
                      f"({sl if sl in banned_set else cl})")
        elif cl in filtered_lower:
            reason = f"capital string '{cap}' occurs in the wash corpus"
        elif sl in filtered_lower:
            reason = f"state string '{state}' occurs in the wash corpus"
        a_ids = tok.encode(" " + state)
        if reason is None and len(a_ids) != 1:
            reason = (f"answer tokenizes to {len(a_ids)} tokens {a_ids} — "
                      f"first-token measurement would not be exact")
        ex = [pool[j] for j in range(len(pool)) if j != i][:e1.K_SHOT]
        prefix = "".join(sent.format(c=c, a=a) for c, a in ex)
        prompt = prefix + query.format(a=cap, c=state)
        rec = {"order": len(probed) + len(dropped), "relation": "tmpl",
               "subject": f"The state whose capital is {cap}",
               "answer": state, "fact": f"{cap}->{state}",
               "prompt": prompt,
               "exemplars": [f"{c}->{a}" for c, a in ex],
               "answer_ids": a_ids}
        if reason is not None:
            rec["drop_reason"] = reason
            dropped.append(rec)
            continue
        rec["ans_id"] = a_ids[0]
        rec["ids"] = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        n_tok = int((train_ids == rec["ans_id"]).sum().item())
        if n_tok > 0:
            rec["drop_reason"] = (f"answer token id {rec['ans_id']} occurs "
                                  f"{n_tok}x in the exact training stream")
            dropped.append(rec)
        else:
            rec["train_token_count"] = n_tok
            probed.append(rec)
    return probed, dropped


# --------------------------------------------------------------- the fresh wash

def fresh_wash(net0, train_ids, ckpt_steps, on_ckpt, resume=None):
    """The FRESH draw of e182's frozen 5e-5 wash, GPU fp32 (TF32 OFF) in
    owner-envelope bursts: AdamW (0.9,0.95) wd 0.1 constant lr 5e-5, clip
    1.0, full-token CE, batch 8 x ctx 512; per step BATCH window offsets
    from the CPU generator seeded FRESH_SEED (e182's device-independent
    bit-identical-draw discipline — the ONLY delta from phase 1's wash).
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
        opt_state_dev_fix(opt, dev)
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
        opt_sd = _opt_state_to_cpu(opt.state_dict())
        torch.save({"model": sd, "opt": opt_sd, "gen": gen.get_state(),
                    "step": step,
                    "meta": {"experiment": NAME, "seed": FRESH_SEED,
                             "lr": e1.LR}}, latest)
        if hit_ckpt is not None:
            torch.save({"model": sd,
                        "meta": {"experiment": NAME, "step": hit_ckpt,
                                 "lr": e1.LR, "seed": FRESH_SEED,
                                 "desc": f"openai-community/gpt2@"
                                 f"{e1.MODEL_REV} e182c2 FRESH draw "
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


def _opt_state_to_cpu(osd: dict) -> dict:
    out = {}
    for k, v in osd.items():
        if k == "state":
            out[k] = {i: {kk: (vv.to("cpu").clone() if torch.is_tensor(vv)
                               else vv) for kk, vv in g.items()}
                      for i, g in v.items()}
        else:
            out[k] = v
    return out


def opt_state_dev_fix(opt, dev):
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if st:
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(dev)


# ------------------------------------------------------------------ plots

def plot_template(rd, tmpl_states, adj1, near_curve, ctrl_curve,
                  fact_curve, form_match):
    """Part 1's figure: the four batteries' retention on phase-1's SAVED
    states + the template table + the verdict."""
    steps = [s["step"] for s in tmpl_states]
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "tab:purple"}
    R0 = {"fact": fact_curve[0], "ctrl": ctrl_curve[0],
          "near": near_curve[0], "tmpl": tmpl_states[0]["tmpl"]["mean_p"]}
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    ax = axes[0, 0]
    N1 = {"fact": 20, "ctrl": 12, "near": 3}
    for b, curve in (("fact", fact_curve), ("ctrl", ctrl_curve),
                     ("near", near_curve)):
        ax.plot(steps, [r / R0[b] for r in curve], "o--", ms=6, lw=1.8,
                color=cols[b], alpha=0.6,
                label=f"{b} (phase-1 record, n={N1[b]})")
    ax.plot(steps, [s["tmpl"]["mean_p"] / R0["tmpl"] for s in tmpl_states],
            "D-", ms=8, lw=2.6, color=cols["tmpl"],
            label=f"TEMPLATE battery (reversed form, NEW, n="
                  f"{len(adj1['tmpl_facts'])})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("phase-1 replay wash steps (SAVED states, CPU fp32)")
    ax.set_ylabel("retention R(s)/R(0)")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    v = adj1["verdict"]
    ax.set_title(f"THE TEMPLATE LOCUS — the reversed form vs the capital-of "
                 f"form on the SAME states -> {v}", fontsize=9.5)

    ax = axes[0, 1]
    xs = steps
    ax.plot(xs, near_curve, "o--", color=cols["near"], alpha=0.6,
            label="nearrel mean_p (capital-of form)")
    ax.plot(xs, [s["tmpl"]["mean_p"] for s in tmpl_states], "D-",
            color=cols["tmpl"], lw=2.4,
            label="template battery mean_p (reversed form)")
    ax.plot(xs, ctrl_curve, "s:", color=cols["ctrl"], alpha=0.6,
            label="ctrl mean_p (generic reference)")
    for x, y in zip(xs, [s["tmpl"]["mean_p"] for s in tmpl_states]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(3, 6), fontsize=6.8, color=cols["tmpl"])
    ax.set_xlabel("wash steps (saved states)")
    ax.set_ylabel("R = battery mean p(answer first token)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    d = adj1["declines_at_deepest"]
    ax.set_title(f"absolute recall — declines at +{d['step']}: tmpl "
                 f"{d['tmpl']:.3f} vs near {d['near']:.3f} "
                 f"(band [{d['band_lo']:.3f}, {d['band_hi']:.3f}]); "
                 f"ctrl {d['ctrl']:.3f}, fact {d['fact']:.3f}", fontsize=9)

    ax = axes[1, 0]
    for b, curve in (("near", near_curve), ("ctrl", ctrl_curve),
                     ("fact", fact_curve)):
        ax.plot(steps, [1 - r / R0[b] for r in curve], "o--", ms=6,
                lw=1.8, color=cols[b], alpha=0.6, label=f"{b} decline")
    ax.plot(steps, [1 - s["tmpl"]["mean_p"] / R0["tmpl"]
                    for s in tmpl_states], "D-", ms=8, lw=2.6,
            color=cols["tmpl"], label="tmpl decline (NEW)")
    d = adj1["declines_at_deepest"]
    ax.axhline(d["band_lo"], color="tab:purple", ls="--", lw=1.3,
               label=f"TEMPLATE-GENERAL edge (near/1.5 = {d['band_lo']:.3f})")
    ax.set_xlabel("wash steps (saved states)")
    ax.set_ylabel("decline 1 - R(s)/R(0)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("declines vs the 1.5x band (below the dashed line = "
                 "TEMPLATE-SPECIFIC)", fontsize=9.5)

    ax = axes[1, 1]
    ax.axis("off")
    y = 0.98
    ax.text(0.02, y, "GATES:", fontsize=9, va="top", family="monospace",
            weight="bold")
    y -= 0.026
    for g, val in adj1["gates_summary"].items():
        ax.text(0.02, y, f"  {g:12s} {'PASS' if val else 'FAIL'}",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.020
    y -= 0.012
    ax.text(0.02, y, "TEMPLATE BATTERY (p0 -> deepest, retention):",
            fontsize=8.6, va="top", family="monospace", weight="bold")
    y -= 0.024
    c0 = {f: v["p"] for f, v in tmpl_states[0]["tmpl"]["probes"].items()}
    cl = {f: v["p"] for f, v in tmpl_states[-1]["tmpl"]["probes"].items()}
    for fact in adj1["tmpl_facts"][:20]:
        ax.text(0.02, y, f"  {fact:30s} {c0[fact]:6.3f} {cl[fact]:7.3f} "
                f"{cl[fact] / c0[fact]:6.2f}",
                fontsize=6.2, va="top", family="monospace")
        y -= 0.0185
        if y < 0.40:
            break
    y = 0.38
    ax.text(0.02, y, f"TEMPLATE VERDICT: {adj1['verdict']}", fontsize=9.4,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.032
    for wd in textwrap.wrap(adj1["clause"], width=92,
                            break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        y -= 0.021
    y -= 0.006
    for wd in textwrap.wrap(form_match, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.1, va="top",
                family="monospace", color="dimgray")
        y -= 0.019

    fig.suptitle("E182C2 PART 1 — THE TEMPLATE CONTROL: the reversed form "
                 f"('The state whose capital is X is', n="
                 f"{len(adj1['tmpl_facts'])}) on phase-1's saved states -> "
                 f"{adj1['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "template_locus.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def plot_fresh(rd, p2, adj2, p1_ref, form_match):
    """Part 2's figure: the fresh draw's four batteries + ppl + the
    phase-1 comparison + the verdict."""
    steps = [s["step"] for s in p2]
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "tab:purple"}
    R0 = {b: p2[0][b]["mean_p"] for b in ("fact", "ctrl", "near", "tmpl")}
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    ax = axes[0, 0]
    N2 = {"fact": 20, "ctrl": 12, "near": 3,
          "tmpl": len(adj2["tmpl_facts"])}
    for b in ("fact", "ctrl", "near", "tmpl"):
        ax.plot(steps, [s[b]["mean_p"] / R0[b] for s in p2], "o-", ms=7,
                lw=2.2, color=cols[b],
                alpha=0.6 if b == "near" else 1.0,
                label=f"{b} retention (n={N2[b]})")
    axr = ax.twinx()
    axr.plot(steps, [s["ppl"] for s in p2], "s:", ms=5, lw=1.4,
             color="seagreen", alpha=0.7)
    axr.set_ylabel("wash-corpus ppl (dotted, right; IMPROVES)", fontsize=8,
                   color="seagreen")
    ax.set_xlabel(f"fresh-draw wash steps (seed {FRESH_SEED}, GPU fp32 "
                  f"bursts)")
    ax.set_ylabel("retention R(s)/R(0)")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"THE FRESH DRAW (n=2 draws now) -> {adj2['verdict']}",
                 fontsize=10)

    ax = axes[0, 1]
    labels = ["fact", "ctrl", "near", "tmpl"]
    x = range(len(labels))
    ph1 = [p1_ref["decl50"].get(b) or 0.0 for b in labels]
    fr = [(adj2.get("declines") or {}).get(b) or 0.0 for b in labels]
    w = 0.38
    ax.bar([i - w / 2 for i in x], ph1, w, color="gray", alpha=0.55,
           label="phase-1 draw (seed 18202, CPU fp32)")
    ax.bar([i + w / 2 for i in x], fr, w, color="tab:orange", alpha=0.85,
           label=f"fresh draw (seed {FRESH_SEED}, GPU fp32)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"decline at +{DRAW_STEP}")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    r = (adj2.get("draw_bars") or {}).get("ratio")
    r_txt = (f"{r:.2f} in [{DRAW_RATIO_LO}, {DRAW_RATIO_HI}]"
             if r is not None
             else "undefined (fact decline at floor)")
    ax.set_title("draw-vs-draw declines at +50: generic ratio ctrl/fact = "
                 + r_txt, fontsize=9)

    ax = axes[1, 0]
    for b in ("fact", "ctrl", "near", "tmpl"):
        ax.plot(steps, [s[b]["frac_top1"] for s in p2], "o-", ms=6,
                lw=1.8, color=cols[b], alpha=0.6 if b == "near" else 1.0,
                label=f"{b}: frac argmax-correct")
    ax.set_xlabel("fresh-draw wash steps")
    ax.set_ylabel("fraction of battery (argmax)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("discrete recall on the fresh draw", fontsize=9.5)

    ax = axes[1, 1]
    ax.axis("off")
    y = 0.98
    ax.text(0.02, y, "FRESH-DRAW TABLE (mean_p per state):", fontsize=9,
            va="top", family="monospace", weight="bold")
    y -= 0.026
    hdr = (f"  {'battery':8s} " + " ".join(f"{f'+{s}':>8s}" for s in steps)
           + "   decl+50")
    ax.text(0.02, y, hdr, fontsize=7.0, va="top", family="monospace")
    y -= 0.022
    for b in ("fact", "ctrl", "near", "tmpl"):
        row = (f"  {b:8s} " + " ".join(f"{s[b]['mean_p']:8.4f}" for s in p2)
               + f"   {adj2['declines'][b]:6.3f}")
        ax.text(0.02, y, row, fontsize=7.0, va="top", family="monospace")
        y -= 0.021
    y -= 0.014
    ax.text(0.02, y, f"DRAW VERDICT: {adj2['verdict']}", fontsize=9.6,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.032
    for wd in textwrap.wrap(adj2["clause"], width=92,
                            break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        y -= 0.021
    y -= 0.010
    for wd in textwrap.wrap(" ".join(adj2["clause_booleans"]),
                            width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.4, va="top",
                family="monospace", color="dimgray")
        y -= 0.019

    fig.suptitle(f"E182C2 PART 2 — THE FRESH CORPUS DRAW (seed "
                 f"{FRESH_SEED}; the only delta from phase-1's wash) -> "
                 f"{adj2['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "fresh_draw.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp1 = rd / "journal_p1.json"
    jp2 = rd / "journal_p2.json"
    log(f"E182C2 PHASE 2 — THE TEMPLATE LOCUS + FRESH DRAWS "
        f"(smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e182c2_template",
        "phase": "2 (e182c's unlock: the template control on the saved "
                 "states + one fresh wash draw)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": ("bars frozen VERBATIM from the dispatch brief; "
                         "fresh seed 20261002 + template pool registered "
                         "and committed BEFORE any wash compute; no bar "
                         "shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("(1) does phase-1's template-level erosion generalize "
                     "across template FORMS (reversed copular) or is it "
                     "capital-of-form-specific? (2) does the phase-1 "
                     "pattern (generic erosion; near-related fastest) "
                     "replicate on a fresh draw?"),
        "builds_on": ["e182c phase 1 / T149 (the unlock + the saved states)",
                      "e182/T123 (the parent wash)",
                      "SUPERVISOR check-in 12 item 3 (the control letter)"],
        "whats_new": ["the reversed-form template battery (the form "
                      "manipulation phase 1 could not make)",
                      "the eval-only re-read of phase-1's SAVED states",
                      "the fresh stream seed 20261002 (n=2 wash draws)",
                      "the template-locus + fresh-draw adjudications"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P0 the records
    p1_metrics_path = common.REPO / "runs" / "e182c" / "metrics.json"
    if not p1_metrics_path.exists():
        log(f"FATAL: phase-1 record missing: {p1_metrics_path}")
        return 1
    p1m = json.loads(p1_metrics_path.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    p1_states = {s["step"]: s for s in p1m["states"]}
    p1_adj = p1m["adjudication"]
    p1_decl = p1_adj["declines_per_state"]           # fact/ctrl per state
    p1_near_decl = p1_adj["nearrel_co_report"]["declines"]
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    if not SMOKE:
        for s_ in SAVED_STATES:
            assert s_ in p1_states, f"phase-1 record lacks state +{s_}"

    # the saved-state provenance inventory (G_STATES part A)
    _ck_name = ("e182c_smoke_s{s}.pt" if SMOKE else "e182c_s{s}.pt")
    ck_files = {s_: common.REPO / "runs" / "checkpoints"
                / _ck_name.format(s=s_) for s_ in SAVED_STATES}
    inv = {
        "saved_states_expected": list(SAVED_STATES),
        "files": {str(s_): {"path": str(p),
                            "exists": p.exists(),
                            "size_bytes": (p.stat().st_size
                                           if p.exists() else None),
                            "mtime": (time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ",
                                time.gmtime(p.stat().st_mtime))
                                if p.exists() else None)}
                  for s_, p in ck_files.items()},
        "phase1_status": p1m.get("status"),
        "finding": ("phase-1's per-state weight archive exists on disk "
                    "(the discipline e182 lacked) — part (1) is EVAL-ONLY "
                    "on these saved states; verified by the G_STATES "
                    "re-probe below"),
    }
    G_STATES_A = {"inventory": inv,
                  "all_exist": all(p.exists() for p in ck_files.values()),
                  "pass": None}   # set after the re-probe
    metrics["inventory"] = inv
    log(f"G_STATES inventory: all saved states on disk = "
        f"{G_STATES_A['all_exist']}")

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
    # the fact battery (needed for answer_ids in the corpus build)
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
                "re-filtered without changing the wash); the fresh draw "
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

    # G_BATT: kept sets + t=0 probes vs phase-1's record
    def _dp_vs_p1(cur_rows, key):
        p1p = p1_states[0][key]["probes"]
        dp = {r["fact"]: abs(r["p"] - p1p[r["fact"]]["p"])
              for r in cur_rows if r["fact"] in p1p}
        return {"kept_set_equal": bool({r["fact"] for r in cur_rows}
                                       == set(p1p)),
                "max_per_probe_dp": max(dp.values()) if dp else None,
                "n": len(dp)}
    G_BATT = {
        "fact": _dp_vs_p1(battery, "fact") if 0 in p1_states else None,
        "ctrl": _dp_vs_p1(cbattery, "ctrl") if 0 in p1_states else None,
        "near": _dp_vs_p1(nbattery, "near") if 0 in p1_states else None,
        "tol_per_probe_dp": TOL_PROBE_DP,
        "note": "the fact/ctrl/nearrel batteries are phase-1's VERBATIM "
                "(module import); t=0 must reproduce phase-1's record "
                "(same pristine organism, same prompts, same CPU fp32 "
                "probe path)",
    }
    G_BATT["pass"] = bool(
        all(G_BATT[k] and G_BATT[k]["kept_set_equal"]
            and G_BATT[k]["max_per_probe_dp"] <= TOL_PROBE_DP
            for k in ("fact", "ctrl", "near"))) if not SMOKE else True
    metrics["gates"] = {"G_SIZE": G_SIZE, "G_CORPUS": G_CORPUS,
                        "G_BATT": G_BATT}
    log(f"G_BATT: fact/ctrl/near kept sets + t=0 probes vs phase-1 record "
        f"-> {'PASS' if G_BATT['pass'] else 'FAIL'} "
        f"(max dp fact {G_BATT['fact']['max_per_probe_dp']}, "
        f"ctrl {G_BATT['ctrl']['max_per_probe_dp']}, "
        f"near {G_BATT['near']['max_per_probe_dp']})")

    # THE TEMPLATE BATTERY (the new instrument)
    tcand, tdropped = build_tmpl_candidates(tok, filtered.lower(),
                                            train_ids, e_banned)
    tbase = e1.probe_battery(net0, tcand)
    for r, b in zip(tcand, tbase["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    tmpl_kept, tsel = e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]
    reduced = len(tbattery) < e1.BATTRY_MIN
    G_TMPL = {
        "template": list(TMPL2),
        "pool_size": len(TMPL_POOL),
        "pool_frozen_in_script": True,
        "gate": tsel,
        "floor": e1.BATTRY_FLOOR,
        "reduced_flag": bool(reduced),
        "drops": [{"fact": r["fact"], "reason": r["drop_reason"]}
                  for r in tdropped],
        "kept_contamination_scans_zero": True,
        "kept_overlap_with_e182_banned_zero": True,
        "multi_token_answers_zero": True,
        "screening": SCREENING_RECORD,
        "exemplar_mechanics": "e182 VERBATIM (pool[0]+pool[1] fixed "
                              "beyond the first two probes)",
        "pass": bool(len(tbattery) >= e1.BATTRY_FLOOR),
    }
    metrics["template_battery"] = {
        "definition": "THE TEMPLATE CONTROL: the capital-of cloze in the "
                      "REVERSED copular form ('The state whose capital is "
                      "{a} is' -> the STATE) over the held-out US-state "
                      "family (disjoint from the fact battery exactly as "
                      "phase-1's nearrel); asks whether phase-1's fast "
                      "template-level erosion tracks the FORM or survives "
                      "the form change",
        "gate": {k: v for k, v in G_TMPL.items() if k != "drops"},
        "kept_facts": tmpl_kept,
        "baseline_all_candidates": [
            {k: r.get(k) for k in ("fact", "relation", "subject", "answer",
                                   "prompt", "p", "rank", "top1", "top5",
                                   "gate_pass", "kept", "drop_reason")}
            for r in tcand],
        "dropped_pre_probe": G_TMPL["drops"],
    }
    log(f"template battery: {len(tbattery)} kept of {len(tcand)} probed "
        f"({tsel['n_pass']} passed the gate)"
        + (f" — FLAGGED reduced (<{e1.BATTRY_MIN})" if reduced else ""))
    assert G_TMPL["pass"] or SMOKE, "template battery below floor"

    # form-matching record (Rule 12)
    f_p0 = [r["p"] for r in battery]
    c_p0 = [r["p"] for r in cbattery]
    n_p0 = [r["p"] for r in nbattery]
    t_p0 = [r["p"] for r in tbattery]
    form_match = (
        "FORM-MATCHING: all batteries 2-shot rotating leave-self-out cloze, "
        f"single-token answers, gate (top1 p>=0.8)|(top5 p>=0.5), cap 20; "
        f"fact n={len(f_p0)} (mean {sum(f_p0)/len(f_p0):.3f}), ctrl n="
        f"{len(c_p0)} (mean {sum(c_p0)/len(c_p0):.3f}), nearrel n="
        f"{len(n_p0)} (mean {sum(n_p0)/len(n_p0):.3f}), template n="
        f"{len(t_p0)} (p0 {min(t_p0):.3f}-{max(t_p0):.3f}, mean "
        f"{sum(t_p0)/len(t_p0):.3f}); the template battery shares the "
        "nearrel/fact CONTENT FAMILY (capitals, held-out entities) but "
        "reverses the template FORM and direction; declines are "
        "per-battery relative.")
    metrics["form_matching"] = form_match
    log(form_match)
    write_metrics("PARTIAL: batteries built; saved-state re-read pending")

    # ------------------------------------------------- P4 part 1: saved states
    def bat_rec(b):
        return {"mean_p": b["mean_p"], "frac_top1": b["frac_top1"],
                "frac_top5": b["frac_top5"],
                "probes": {r["fact"]: {"p": r["p"], "rank": r["rank"]}
                           for r in b["probes"]}}

    p1_new = []      # the template battery on the saved states
    state_dps = []   # G_STATES re-probe deltas
    if jp1.exists():
        try:
            p1_new = json.loads(jp1.read_text(encoding="utf-8"))["states"]
            log(f"journal_p1: {len(p1_new)} states restored")
        except Exception as e:  # noqa: BLE001
            log(f"journal_p1 unreadable ({e}); recomputing")
            p1_new = []
    done1 = {s["step"] for s in p1_new}

    # t=0 on the pristine organism
    if 0 not in done1:
        tb = e1.probe_battery(net0, tbattery)
        p1_new.append({"step": 0, "tmpl": bat_rec(tb)})
        p1_new.sort(key=lambda s: s["step"])
    for s_ in SAVED_STATES:
        if s_ in done1:
            continue
        f = ck_files[s_]
        sd = torch.load(f, map_location=CPU, weights_only=False)["model"]
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        tb = e1.probe_battery(evl, tbattery)
        rec = {"step": s_, "tmpl": bat_rec(tb)}
        del evl, sd
        p1_new.append(rec)
        p1_new.sort(key=lambda s: s["step"])
        jp1.write_text(json.dumps({"states": p1_new}, indent=1),
                       encoding="utf-8")
        log(f"  SAVED STATE +{s_}: template battery "
            f"{tb['mean_p']:.4f} (ret "
            f"{tb['mean_p'] / p1_new[0]['tmpl']['mean_p']:.3f})"
            + (f"; G_STATES re-probe max dp "
               f"{max(state_dps) if state_dps else 0:.5f}"
               if state_dps else ""))
        metrics["part1_template"] = {"states": p1_new}
        write_metrics(f"PARTIAL: part 1 through saved state +{s_}")

    # G_STATES re-probe pass (also fills reprobes missing after a journal
    # resume): ctrl+nearrel on the LOADED s50/s80 vs phase-1's record
    st1_pre = {s["step"]: s for s in p1_new}
    for s_ in ((50, 80) if not SMOKE else ()):
        if s_ not in st1_pre or "ctrl_reprobe" in st1_pre[s_]:
            continue
        sd = torch.load(ck_files[s_], map_location=CPU,
                        weights_only=False)["model"]
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        cb = e1.probe_battery(evl, cbattery)
        nb = e1.probe_battery(evl, nbattery)
        st1_pre[s_]["ctrl_reprobe"] = bat_rec(cb)
        st1_pre[s_]["near_reprobe"] = bat_rec(nb)
        del evl, sd
        jp1.write_text(json.dumps({"states": p1_new}, indent=1),
                       encoding="utf-8")
    for s_ in ((50, 80) if not SMOKE else ()):
        for key in ("ctrl_reprobe", "near_reprobe"):
            if s_ not in st1_pre or key not in st1_pre[s_]:
                continue
            p1key = key.split("_")[0]
            p1p = p1_states[s_][p1key]["probes"]
            for fact, v in st1_pre[s_][key]["probes"].items():
                state_dps.append(abs(v["p"] - p1p[fact]["p"]))
    if state_dps:
        log(f"G_STATES re-probe: {len(state_dps)} probes, max dp "
            f"{max(state_dps):.6f} (tol {TOL_STATE_DP})")
    G_STATES = {**G_STATES_A,
                "reprobe_max_dp": max(state_dps) if state_dps else None,
                "tol_reprobe_dp": TOL_STATE_DP,
                "reprobe_note": "ctrl+nearrel re-probed on the LOADED s50/"
                                "s80 and compared per-probe to phase-1's "
                                "record (same fp32 weights + same CPU fp32 "
                                "probe path -> expected ~0.0)"}
    G_STATES["pass"] = bool(G_STATES_A["all_exist"]
                            and (SMOKE or (state_dps
                                           and max(state_dps)
                                           <= TOL_STATE_DP)))
    metrics["gates"]["G_STATES"] = G_STATES
    metrics["part1_template"] = {"states": p1_new}
    write_metrics("PARTIAL: part 1 states read; adjudication pending")

    # ------------------------------------------ P5 part-1 adjudication (frozen)
    st1 = {s["step"]: s for s in p1_new}
    R0t = st1[0]["tmpl"]["mean_p"]
    deepest1 = max(st1)
    tmpl_decl = {str(s["step"]): 1 - s["tmpl"]["mean_p"] / R0t
                 for s in p1_new if s["step"] > 0}

    # the phase-1 curves, RUNTIME-READ (steps 0 + SAVED_STATES, filtered to
    # the steps phase-1's record actually contains — smoke uses {1,2})
    grid = [0] + [s for s in SAVED_STATES if s in p1_states]
    near_curve = [p1_states[s]["near"]["mean_p"] for s in grid]
    ctrl_curve = [p1_states[s]["ctrl"]["mean_p"] for s in grid]
    fact_curve = [p1_states[s]["fact"]["mean_p"] for s in grid]
    near80 = 1 - p1_states[deepest1]["near"]["mean_p"] / near_curve[0]
    ctrl80 = 1 - p1_states[deepest1]["ctrl"]["mean_p"] / ctrl_curve[0]
    fact80 = 1 - p1_states[deepest1]["fact"]["mean_p"] / fact_curve[0]
    t80 = tmpl_decl.get(str(deepest1))
    band_lo, band_hi = near80 / TEMPLATE_BAND, near80 * TEMPLATE_BAND

    gates_ok_1 = bool(G_STATES["pass"] and G_CORPUS["pass"]
                      and G_BATT["pass"] and G_TMPL["pass"])
    if SMOKE:
        verdict1, clause1 = "SMOKE (nothing adjudicated)", "smoke run"
        tg, ts_ = None, None
    elif not gates_ok_1:
        verdict1 = "VERIFICATION-FAILED (curves reported; no bar read)"
        clause1 = ("verification gates failed: "
                   + ", ".join(g for g, v in (("G_STATES", G_STATES["pass"]),
                                              ("G_CORPUS", G_CORPUS["pass"]),
                                              ("G_BATT", G_BATT["pass"]),
                                              ("G_TMPL", G_TMPL["pass"]))
                               if not v))
        tg, ts_ = None, None
    else:
        tg = bool(near80 > DECL_FLOOR and band_lo <= t80 <= band_hi)
        ts_ = bool(near80 > DECL_FLOOR and t80 < band_lo)
        if tg:
            verdict1 = "TEMPLATE-GENERAL"
            clause1 = (f"the second template's controls erode comparably "
                       f"to the first's at +{deepest1} (reversed-form "
                       f"decline {t80:.3f} vs capital-of-form nearrel "
                       f"{near80:.3f}, ratio {t80 / near80:.2f}; band "
                       f"[{band_lo:.3f}, {band_hi:.3f}]) — the erosion is "
                       "template-GENERAL; the locus is the few-shot-"
                       "following faculty, not one form (with the disclosed "
                       "caveat: form and direction changed together)")
        elif ts_:
            verdict1 = "TEMPLATE-SPECIFIC"
            clause1 = (f"the second template's controls hold materially "
                       f"better at +{deepest1} (reversed-form decline "
                       f"{t80:.3f} < band edge {band_lo:.3f} = nearrel "
                       f"{near80:.3f}/1.5; generic-reference ctrl "
                       f"{ctrl80:.3f}) — the erosion is template-specific; "
                       "the hint rescopes to the capital-of form")
        else:
            verdict1 = "GRADED"
            clause1 = (f"partial: the reversed form erodes FASTER than the "
                       f"band at +{deepest1} (decline {t80:.3f} > "
                       f"{band_hi:.3f} = nearrel {near80:.3f}*1.5) — "
                       "neither registered template bar fires; the tables "
                       "verbatim")
    # +50 co-adjudication
    co = None
    if 50 in st1 and not SMOKE and gates_ok_1:
        near50 = 1 - p1_states[50]["near"]["mean_p"] / near_curve[0]
        t50 = tmpl_decl.get("50")
        co = {"step": 50, "tmpl": t50, "near": near50,
              "band": [near50 / TEMPLATE_BAND, near50 * TEMPLATE_BAND],
              "general": bool(near50 > DECL_FLOOR
                              and near50 / TEMPLATE_BAND <= t50
                              <= near50 * TEMPLATE_BAND),
              "specific": bool(near50 > DECL_FLOOR
                               and t50 < near50 / TEMPLATE_BAND)}
    metrics["adjudication_part1_template"] = {
        "bars": REGISTERED_PREDICTION["bars_verbatim"],
        "TEMPLATE_GENERAL": tg, "TEMPLATE_SPECIFIC": ts_,
        "verdict": verdict1, "clause": clause1,
        "bar_constants": {"TEMPLATE_BAND": TEMPLATE_BAND,
                          "DECL_FLOOR": DECL_FLOOR},
        "declines_at_deepest": {"step": deepest1, "tmpl": t80,
                                "near": near80, "ctrl": ctrl80,
                                "fact": fact80,
                                "band_lo": band_lo, "band_hi": band_hi,
                                "ratio_tmpl_over_near":
                                    (t80 / near80
                                     if near80 > 1e-9 else None),
                                "ratio_tmpl_over_ctrl":
                                    (t80 / ctrl80
                                     if abs(ctrl80) > 1e-9 else None)},
        "tmpl_declines_per_state": tmpl_decl,
        "phase1_reference_declines": {
            "note": "runtime-read from runs/e182c/metrics.json",
            "near": p1_near_decl, "fact_ctrl": p1_decl},
        "co_adjudication_at_50": co,
        "gates_summary": {"G_STATES": G_STATES["pass"],
                          "G_CORPUS": G_CORPUS["pass"],
                          "G_BATT": G_BATT["pass"],
                          "G_TMPL": G_TMPL["pass"]},
        "tmpl_facts": [r["fact"] for r in tbattery],
    }
    log("=" * 78)
    log(f"PART 1 TEMPLATE VERDICT: {verdict1}")
    log(f"  {clause1}")
    write_metrics("PARTIAL: part 1 adjudicated; fresh draw pending")

    # ---------------------------------------------- P6 part 2: the fresh draw
    latest_ck = (common.REPO / "runs" / "checkpoints"
                 / f"{NAME}_fresh_latest.pt")
    p2_states = []
    resume = None
    if jp2.exists():
        try:
            p2_states = json.loads(jp2.read_text(encoding="utf-8"))["states"]
            log(f"journal_p2: {len(p2_states)} states restored")
        except Exception as e:  # noqa: BLE001
            log(f"journal_p2 unreadable ({e}); fresh")
            p2_states = []
    if latest_ck.exists():
        try:
            resume = torch.load(latest_ck, map_location=CPU,
                                weights_only=False)
            assert max(s["step"] for s in p2_states) == resume["step"], \
                "journal/checkpoint step mismatch"
        except Exception as e:  # noqa: BLE001
            log(f"resume failed ({e}); fresh wash from scratch")
            resume = None
            p2_states = [s for s in p2_states if s["step"] == 0]

    def state_rec2(step, fb, cb, tb2, nb, hp, ce=None):
        return {"step": step, "fact": bat_rec(fb), "ctrl": bat_rec(cb),
                "tmpl": bat_rec(tb2), "near": bat_rec(nb),
                "ppl": hp["ppl"], "ce": hp["ce"], "in_batch_ce": ce}

    if 0 not in {s["step"] for s in p2_states}:
        fb = e1.probe_battery(net0, battery)
        cb = e1.probe_battery(net0, cbattery)
        tb2 = e1.probe_battery(net0, tbattery)
        nb = e1.probe_battery(net0, nbattery)
        hp = e1.ppl_eval(net0, *bank_xy)
        p2_states.append(state_rec2(0, fb, cb, tb2, nb, hp))
        p2_states.sort(key=lambda s: s["step"])
        jp2.write_text(json.dumps({"states": p2_states}, indent=1),
                       encoding="utf-8")
        log(f"fresh t=0: fact {fb['mean_p']:.4f} | ctrl {cb['mean_p']:.4f} "
            f"| tmpl {tb2['mean_p']:.4f} | near {nb['mean_p']:.4f} | ppl "
            f"{hp['ppl']:.2f}")
        metrics["part2_fresh"] = {"states": p2_states}
        write_metrics("PARTIAL: fresh t=0 read; wash pending")

    base_f = {b: p2_states[0][b]["mean_p"]
              for b in ("fact", "ctrl", "tmpl", "near")}

    def on_ckpt2(step, sd, ce):
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        fb = e1.probe_battery(evl, battery)
        cb = e1.probe_battery(evl, cbattery)
        tb2 = e1.probe_battery(evl, tbattery)
        nb = e1.probe_battery(evl, nbattery)
        hp = e1.ppl_eval(evl, *bank_xy)
        del evl
        rec = state_rec2(step, fb, cb, tb2, nb, hp, ce)
        p2_states.append(rec)
        p2_states.sort(key=lambda s: s["step"])
        jp2.write_text(json.dumps({"states": p2_states}, indent=1),
                       encoding="utf-8")
        log(f"  FRESH CKPT +{step:3d}: fact {fb['mean_p']:.4f} (decl "
            f"{1 - fb['mean_p'] / base_f['fact']:.3f}) | ctrl "
            f"{cb['mean_p']:.4f} (decl "
            f"{1 - cb['mean_p'] / base_f['ctrl']:.3f}) | tmpl "
            f"{tb2['mean_p']:.4f} (decl "
            f"{1 - tb2['mean_p'] / base_f['tmpl']:.3f}) | near "
            f"{nb['mean_p']:.4f} (decl "
            f"{1 - nb['mean_p'] / base_f['near']:.3f}) | ppl "
            f"{hp['ppl']:.2f} | in-batch CE {ce:.4f}")
        metrics["part2_fresh"] = {"states": p2_states}
        write_metrics(f"PARTIAL: fresh draw through +{step}")

    todo2 = tuple(s for s in FRESH_CK
                  if s > max(s2["step"] for s2 in p2_states))
    envelope_poll_log = []
    if todo2:
        log(f"THE FRESH DRAW: lr {e1.LR}, {todo2[-1]} steps, batch "
            f"{e1.BATCH} x ctx {e1.SEQ}, AdamW (0.9,0.95) wd 0.1 clip 1.0, "
            f"stream seed {FRESH_SEED} (the ONLY delta from phase-1's "
            f"18202), checkpoints +{list(todo2)}, GPU fp32 bursts "
            f"<={BURST_MAX_STEPS} steps/<={BURST_MAX_S:.0f}s, cooldown "
            f">={COOLDOWN_S:.0f}s, launches double-polled "
            f"util<={LAUNCH_UTIL:.0f}%/temp<={LAUNCH_TEMP:.0f}C")
        fin, envelope_poll_log = fresh_wash(net0, train_ids, todo2,
                                            on_ckpt2, resume=resume)
        log(f"fresh wash finished at step {fin}")
    metrics["envelope"] = {
        "launch_gate": f"double-poll util<={LAUNCH_UTIL:.0f}% AND "
                       f"temp<={LAUNCH_TEMP:.0f}C, {POLL_GAP_S:.0f}s apart",
        "burst_caps": {"wall_s": BURST_MAX_S, "steps": BURST_MAX_STEPS},
        "cooldown_s": COOLDOWN_S,
        "polls_logged": "run log + runs/_envelope_log.jsonl (every poll)",
        "n_launch_poll_cycles": len(envelope_poll_log) // 2,
    }

    # ------------------------------------------ P7 part-2 adjudication (frozen)
    st2 = {s["step"]: s for s in p2_states}
    decl_fresh = {str(s["step"]): {b: 1 - s[b]["mean_p"] / base_f[b]
                                   for b in ("fact", "ctrl", "tmpl", "near")}
                  for s in p2_states if s["step"] > 0}
    G_PPL = {"rule": "bank ppl (the SAME frozen held-out tail bank) read "
                     "at every fresh state",
             "ppl_curve": {str(s["step"]): s["ppl"] for s in p2_states},
             "ppl_improves": bool(p2_states[-1]["ppl"] < p2_states[0]["ppl"]),
             "pass": bool(p2_states[-1]["ppl"] < p2_states[0]["ppl"])}
    metrics["gates"]["G_PPL"] = G_PPL

    d_star = decl_fresh.get(str(DRAW_STEP))
    d_deep = decl_fresh.get(str(DRAW_STEP_DEEP))
    p1_ref = {"decl50": {"fact": p1_decl["50"]["fact"],
                         "ctrl": p1_decl["50"]["ctrl"],
                         "near": p1_near_decl["50"],
                         "tmpl": (co["tmpl"] if co else None)},
              "note": "tmpl = the template battery's decline on PHASE-1's "
                      "saved states at +50 (this cell's part-1 read; "
                      "phase 1 had no reversed-form battery)"}

    def draw_bars(d, step):
        if d is None:
            return None
        fd, cd = d["fact"], d["ctrl"]
        if fd > DECL_FLOOR:
            ratio = cd / fd
            A = bool(DRAW_RATIO_LO <= ratio <= DRAW_RATIO_HI)
        else:
            ratio, A = None, False
        B = bool(d["near"] > fd and d["near"] > cd)
        if A and B:
            v = "DRAW-REPLICATES"
        elif (not A) and (not B):
            v = "DRAW-DIFFERS"
        else:
            v = "GRADED"
        return {"step": step, "ratio": ratio, "clause_A": A, "clause_B": B,
                "verdict": v}

    if SMOKE:
        verdict2, clause2, b2, b2_deep = ("SMOKE (nothing adjudicated)",
                                          "smoke run", None, None)
    elif d_star is None:
        verdict2 = "VERIFICATION-FAILED (fresh draw did not reach +50; " \
                   "curves reported; no bar read)"
        clause2 = (f"deepest fresh state is "
                   f"+{max(s['step'] for s in p2_states)}; the DRAW bars "
                   "adjudicate at +50")
        b2, b2_deep = None, None
    elif not gates_ok_1 or not G_PPL["pass"]:
        verdict2 = "VERIFICATION-FAILED (curves reported; no bar read)"
        clause2 = "part-1 gates failed; the fresh-draw bars are not read"
        b2, b2_deep = None, None
    else:
        b2 = draw_bars(d_star, DRAW_STEP)
        b2_deep = draw_bars(d_deep, DRAW_STEP_DEEP)
        verdict2 = b2["verdict"]
        ratio_txt = (f"{b2['ratio']:.2f}" if b2["ratio"] is not None
                     else "undefined (fact decline at floor)")
        tmpl_note = ("; the template battery outpaces nearrel on this draw"
                     if d_star and d_star["tmpl"] > d_star["near"] else "")
        if verdict2 == "DRAW-REPLICATES":
            clause2 = (f"the fresh draw reproduces the phase-1 pattern at "
                       f"+{DRAW_STEP} (generic erosion ratio ctrl/fact "
                       f"{ratio_txt} in [0.7, 1.3]; the near-related still "
                       f"fastest: nearrel decline {d_star['near']:.3f} > "
                       f"fact {d_star['fact']:.3f}, ctrl {d_star['ctrl']:.3f}"
                       f"{tmpl_note}) — the texture at n=2 draws")
        elif verdict2 == "DRAW-DIFFERS":
            clause2 = (f"the fresh draw's pattern differs materially at "
                       f"+{DRAW_STEP} (ratio {ratio_txt}; clause A "
                       f"{'holds' if b2['clause_A'] else 'fails'}; the "
                       f"near-related-fastest clause B "
                       f"{'holds' if b2['clause_B'] else 'fails'}: nearrel "
                       f"{d_star['near']:.3f} vs fact {d_star['fact']:.3f}, "
                       f"ctrl {d_star['ctrl']:.3f}, tmpl "
                       f"{d_star['tmpl']:.3f}) — the phase-1 texture was "
                       "one draw's; reported honestly")
        else:
            clause2 = (f"partial at +{DRAW_STEP}: clause A (ratio "
                       f"{ratio_txt} in [0.7,1.3]) "
                       f"{'holds' if b2['clause_A'] else 'fails'}; clause B "
                       f"(near-related fastest: nearrel {d_star['near']:.3f}"
                       f" vs fact {d_star['fact']:.3f}, ctrl "
                       f"{d_star['ctrl']:.3f}, tmpl {d_star['tmpl']:.3f}) "
                       f"{'holds' if b2['clause_B'] else 'fails'} — the "
                       "tables verbatim")
    metrics["adjudication_part2_fresh"] = {
        "bars": REGISTERED_PREDICTION["bars_verbatim"],
        "verdict": verdict2, "clause": clause2,
        "draw_bars": b2, "draw_bars_deep": b2_deep,
        "clause_booleans": [f"A (ratio in [0.7,1.3]) = "
                            f"{b2['clause_A'] if b2 else 'n/a'}",
                            f"B (near-related fastest) = "
                            f"{b2['clause_B'] if b2 else 'n/a'}"],
        "bar_constants": {"DRAW_RATIO": [DRAW_RATIO_LO, DRAW_RATIO_HI],
                          "DECL_FLOOR": DECL_FLOOR,
                          "adjudicated_at": DRAW_STEP,
                          "co_adjudicated_at": DRAW_STEP_DEEP},
        "declines": d_star,
        "declines_per_state": decl_fresh,
        "co_adjudication_deep": b2_deep,
        "phase1_reference": p1_ref,
        "gates_summary": {**metrics["adjudication_part1_template"]
                          ["gates_summary"], "G_PPL": G_PPL["pass"]},
        "tmpl_facts": [r["fact"] for r in tbattery],
        "ctrl_facts": [r["fact"] for r in cbattery],
        "fact_facts": [r["fact"] for r in battery],
        "near_facts": [r["fact"] for r in nbattery],
    }
    log("=" * 78)
    log(f"PART 2 FRESH-DRAW VERDICT: {verdict2}")
    log(f"  {clause2}")

    # ------------------------------------------------------------ the honesty
    metrics["honesty_reflex"] = {
        "n_counts": "n=1 template pool (hand-curated, screened, frozen); "
                    "nearrel n=3; n=2 wash draws total (18202 CPU fp32 + "
                    f"{FRESH_SEED} GPU fp32); single organism",
        "form_direction_confound": "the reversed form changes surface form "
            "AND direction together (the dispatch's example form); a "
            "TEMPLATE-GENERAL fire is form-robustness, not proof the "
            "faculty-vs-direction locus is resolved",
        "device_texture": "part (1)'s states are phase-1's CPU fp32 "
            "replay (its own G_REPLAY stamp: <= 0.001 fact mean_p dev "
            "from e182's GPU original at shared steps); part (2)'s wash "
            "is GPU fp32 (TF32 OFF) — the draw bars compare patterns, "
            "never bit values",
        "probes": "CPU fp32 everywhere (phase-1's instrument, "
                  "bit-identical path); 124M inference cost ~54 prompts + "
                  "24x512 bank per state, walls logged per state",
        "nothing_guaranteed": "controls holding does not prove the facts "
            "special; controls eroding does not prove generic; the "
            "openness is the point",
        "fresh_draw_is_texture": "n=2 draws is texture, not law; the "
            "phase-2 promotion owes further draws its generality",
        "first_battery_n3": "the capital-of-form reference (nearrel) is a "
            "3-item battery — its 0.766 decline carries item-level noise "
            "the 19-item second battery does not; the tables verbatim",
        "corpus_frozen": "the wash corpus is INHERITED FROZEN; 'fresh "
            "draw' = fresh window-draw stream (new seed), same filtered "
            "corpus — a fresh CORPUS is a different (future) axis",
    }
    metrics["compute"] = {
        "part1": "eval-only on phase-1's saved states (CPU fp32; no wash "
                 "re-run)",
        "part2": f"GPU fp32 fresh wash, {FRESH_STEPS} steps in owner-"
                 f"envelope bursts (<= {BURST_MAX_STEPS} steps / <= "
                 f"{BURST_MAX_S:.0f}s), cooldown >= {COOLDOWN_S:.0f}s, "
                 "double-polled launches; probes CPU fp32",
        "state_archive": [f"runs/checkpoints/{NAME}_fresh_s{s}.pt"
                          for s in FRESH_CK],
        "resumable": f"runs/checkpoints/{NAME}_fresh_latest.pt",
        "run_log": str(LOG_PATH),
    }
    metrics["trims"] = []
    metrics["deviations"] = deviations
    write_metrics("DONE" if not SMOKE else "SMOKE DONE")

    # ---------------------------------------------------------------- the plots
    pngs = []
    if not SMOKE:
        pngs.append(plot_template(rd, p1_new,
                                  metrics["adjudication_part1_template"],
                                  near_curve, ctrl_curve, fact_curve,
                                  form_match))
        pngs.append(plot_fresh(rd, p2_states,
                               metrics["adjudication_part2_fresh"],
                               p1_ref, form_match))
    log(f"outputs: {rd / 'metrics.json'}, {pngs}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
