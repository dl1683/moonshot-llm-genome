"""E182 — THE GPT-2 WASH: the external-validity fuse, the arc's one true
remaining inverter (T114/T119).

WHY: the lab's no-basin finding lives entirely on 0.84-2.7M char-level nets
grown in-house. T114: the consolidated memory has NO robustness basin — any
AdamW step of the wash's size ends it (content-free noise kills identically);
the corpus's addition is SURGERY (the fact dies, the organism recovers).
T119: the exit is a rate law — t* ~ lr^-1.1..-1.4, basin width ~2.5-5 L2 over
2.7M params; gentle training forgives, ordinary training exits. EVERY clause
of that mechanism paragraph is a claim about the lab's own organisms. THE
QUESTION this cell is the only remaining inverter of: does the field's own
organism (pretrained GPT-2 124M, the field's standard small LM) show the same
physics, or is a real model's factual recall RESISTANT to a plain-corpus
wash? If the lab's physics replicates cross-scale, the paper's mechanism
sentence survives external validity; if GPT-2 resists, the finding is
bounded to the small-net regime and W019's field-facing line dies its final
death.

REGISTERED BARS (frozen here before compute; the dispatch's registration
verbatim; no bar shopping — adjudicate against exactly this):
  - TWO-STEP-WASH fires if: recall drops < 50% of baseline by <= 10 steps at
    healthy perplexity (within 2x its start) — the no-basin physics
    replicates at 124M scale; the finding goes cross-scale.
  - RESISTANT fires if: recall holds >= 80% of baseline through 200 steps at
    the moderate lr — pretrained facts are wash-resistant; the finding is
    bounded to the lab's small-net regime; W019's field-facing line dies its
    final death.
  - TEXTURE in between (slow decay, lr-sensitivity) => TEXTURE with the
    curves.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * RECALL R(c) = the battery's MEAN p(answer first token) over the KEPT
    probes at checkpoint c — the direct analog of the lab's install-60
    battery mean p(Z) (one absolute ruler, the same frozen prompts at every
    checkpoint); retention(c) = R(c)/R(0). Co-reports: frac_top1, frac_top5,
    per-probe retention, the perplexity curve, the in-batch corpus CE.
  * "recall drops < 50% of baseline by <= 10 steps" = retention < 0.5 at a
    MEASURED checkpoint in {2, 10} (this run's grid); "at healthy
    perplexity (within 2x its start)" = the wash-corpus perplexity at that
    checkpoint <= 2x its step-0 value (a wrecked model forgetting is not
    the finding).
  * "recall holds >= 80% of baseline through 200 steps at the moderate lr"
    = the lr 5e-5 arm's retention >= 0.8 at EVERY measured checkpoint
    {2, 10, 50, 200}, with the same perplexity health required at every
    checkpoint (a wrecked model cannot claim resistance either).
  * Adjudication order: TWO-STEP-WASH -> RESISTANT -> TEXTURE; both bars'
    booleans reported regardless. (The clauses are near-disjoint by physics
    — a faster-washing arm cannot also hold — but the order is frozen for
    the pathological corner where the gentle arm washes first.)
  * THE PROBE GATE (selection, frozen): the candidate pool is 36 standard
    cloze facts (18 capitals + 12 languages + 6 currencies, mission's
    exemplar pattern "The capital of France is ___"); a candidate is
    DROPPED before probing if its answer does not tokenize to a single
    GPT-2 token (first-token measurement stays exact); a probed candidate
    is KEPT iff baseline (rank==0 AND p >= 0.8) OR (rank < 5 AND p >= 0.5)
    — the mission's ">= 0.8 top-1 or the token in top-5 with high prob",
    high prob := 0.5, frozen. If > 20 pass, keep the top 20 by baseline p
    (ties by candidate order) — the mission's 10-20 window. The battery
    floor is 10 (a 6-9 battery runs FLAGGED reduced; < 6 aborts).
  * THE FEW-SHOT INSTRUMENT (frozen): each probe's prompt is 2 exemplar
    sentences from its own relation followed by the query (rotating
    leave-self-out, frozen candidate order; a probe's answer NEVER appears
    in its own prompt). Rationale, recorded as a deviation: bare zero-shot
    cloze on GPT-2 124M tops out near p~0.2 on these facts (pilot, this
    session — 1/36 bare candidates over the 0.8 bar); the 2-shot pattern
    is the GPT-2-era standard probing form and puts the battery where the
    mission's gate needs it (0.5-0.95, rank 0). The fixed context is part
    of the frozen instrument: decay under the wash is real, but its LOCUS
    (fact storage vs in-context task-following) is NOT adjudicated here.
  * THE HEALTH GUARD: wash-corpus perplexity on a FIXED held-out bank (the
    corpus tail, never sampled by training) + the in-batch training CE.

DESIGN: (A) probe selection per the frozen gate on the pristine organism
(every candidate's baseline recorded — kept and dropped); (B) THE WASH: the
lab's wash optimizer VERBATIM (AdamW (0.9, 0.95) wd 0.1, constant lr, grad
clip 1.0, full-token CE, CPU-generator window draws) on the lab's plain
corpus (data/input.txt, Shakespeare) LINE-FILTERED so that ZERO occurrences
of any candidate probe string / critical token remain — subject words AND
answer words, case-insensitive, plus a token-id scan of the exact training
stream for every kept answer token; lr {5e-6, 5e-5} (gentle / moderate for a
pretrained 124M), 200 steps, batch 8 x ctx 512; recall checkpoints at
{2, 10, 50, 200} with the perplexity guard; (C) the recall-vs-steps curve
per lr with the perplexity overlay, adjudicated against the bars.

ORGANISM: openai-community/gpt2 @ revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
(124,439,808 params; GPT2TokenizerFast pinned to the same snapshot; loaded
offline from the local HF cache). SIZE ENVELOPE: > the lab's 100M free tier,
within the <=500M ceiling WITH STATED REASON (common.py's envelope): the
reason is EXTERNAL VALIDITY — the mission requires the field's own standard
organism, and GPT-2 124M is the field's standard small pretrained LM; no
smaller pretrained model answers the cross-scale question. Dropout disabled
(all nn.Dropout p=0: the wash under test is the corpus-gradient pressure,
not stochastic noise — e185 already established noise kills; this cell
isolates the stream).

COMPUTE ENVELOPE (dispatch): GPU ALLOWED (idle 0%/62C, 476G free at
dispatch) — strict pre-training quick check per training (gpu_status()/
gpu_ok(): util <= 85 AND temp <= 80 C, plus the mem-headroom guard <= 85%
of total; double-poll 5 s apart), PARK-ONCE to CPU on any failure (no
re-probing, never contention with a returning user; e152R's policy via
e184 verbatim). MID-RUN contention guard: every 25 steps of a GPU training,
re-poll; mem > 85% of total or temp > 80 C -> migrate net + optimizer state
to CPU and FINISH THERE (e184/e152 precedent; any device mixing recorded).
cooldown(90 s) before and after EACH training (dispatch: 60-120 s); caps
1800 s per training (dispatch); ALL readouts CPU-side (battery + perplexity
banks evaluated on a CPU twin); sequential; NO concurrent GPU. Torch
threads 8 (e152R/e143/e184 convention). Final checkpoints saved per arm to
runs/checkpoints/e182_*.pt (gitignored; disk fine per dispatch).

INSTRUMENT PROVENANCE: pick_dev / migrate_to_cpu are lab/e184_seed_
replicates.py VERBATIM (e152R's park-once policy + the mid-run guard);
gpu_status/gpu_ok/cooldown/run_dir/save_json are lab/common.py VERBATIM;
the wash loop is e176n/e180's finetune_freeze arithmetic adapted to a
HF GPT2LMHeadModel (same optimizer, same per-step CPU-generator draw
discipline: torch.randint(hi, (batch,), generator=gen) — device-
independent); the probe battery is the field's standard cloze probe (p of
the answer's first token at the blank), the external analog of the lab's
battery_cell mean p(target-at-last-position); the corpus line-filter +
verification is new but follows e170's neutral-bank rejection logic (plain
corpus minus probe content) at corpus scale. Copied, not imported, to own
the device policy and the organism swap.

NETS: the pristine pretrained organism (revision-pinned, offline cache);
two wash arms from bit-identical pristine copies; final +200 checkpoints
runs/checkpoints/e182_gpt2_lr{5e6,5e5}.pt. NO lab checkpoints touched.

Outputs: runs/e182/{metrics.json, gpt2_wash.png}. No NOTES/THINKING/QUEUE/
STATE edits (dispatch); single commit, no push.

Run:  cd lab && python e182_gpt2_wash.py    (E182_SMOKE=1 shakedown)
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # pinned revision, local cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e152R/e143/e184 convention

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import cooldown, gpu_ok, gpu_status, run_dir, save_json  # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E182_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- organism -----------------------------------------------------------------
MODEL_REPO = "openai-community/gpt2"
MODEL_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SIZE_CEILING = 500_000_000          # common.py's MAX_MODEL_PARAMS_CEILING
SIZE_REASON = ("external validity (dispatch): the field's own standard "
               "organism is required — GPT-2 124M, > the 100M free tier, "
               "<= the 500M ceiling with this stated reason")

# ---- the wash envelope ----------------------------------------------------------
LRS: tuple[float, ...] = (5e-6, 5e-5)     # dispatch: gentle then moderate
LR_TAG = {5e-6: "lr5e6", 5e-5: "lr5e5"}
SEQ = 512                                 # GPT-2's native context
BATCH = 8                                 # dispatch: batch small
STEPS = 4 if SMOKE else 200               # dispatch: 200 steps
CK_MAIN: tuple[int, ...] = (2, 4) if SMOKE else (2, 10, 50, 200)
FREEZE_SEED = 18202                       # wash window draws (CPU generator)
BANK_SEED = 18203                         # (unused reserves; bank is the tail)
TRAIN_CAP_S = 1800.0                      # dispatch: <=1800 s caps
COOLDOWN_S = 90.0                         # dispatch: 60-120 s
MIDRUN_POLL_EVERY = 25                    # mid-run GPU contention poll cadence
BANK_WINDOWS = 6 if SMOKE else 24         # held-out tail windows x SEQ

# ---- the probe battery (frozen candidate pool; mission's exemplar pattern) ------
K_SHOT = 2
GATE_TOP1_P = 0.8                         # ">= 0.8 top-1"
GATE_TOP5_P = 0.5                         # "in top-5 with high prob" (high := 0.5)
BATTRY_MIN, BATTRY_MAX = 10, 20           # mission: 10-20 kept facts
BATTRY_FLOOR = 6                          # below this: abort (cannot support any bar)

CAPS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
        ("Spain", "Madrid"), ("Japan", "Tokyo"), ("England", "London"),
        ("Greece", "Athens"), ("Poland", "Warsaw"), ("Portugal", "Lisbon"),
        ("Egypt", "Cairo"), ("Russia", "Moscow"), ("China", "Beijing"),
        ("Norway", "Oslo"), ("Austria", "Vienna"), ("Ireland", "Dublin"),
        ("Denmark", "Copenhagen"), ("Scotland", "Edinburgh"),
        ("Korea", "Seoul")]
LANGS = [("France", "French"), ("Germany", "German"), ("Japan", "Japanese"),
         ("Italy", "Italian"), ("Spain", "Spanish"), ("Brazil", "Portuguese"),
         ("Russia", "Russian"), ("China", "Chinese"), ("England", "English"),
         ("Greece", "Greek"), ("Egypt", "Arabic"), ("Poland", "Polish")]
CURS = [("Japan", "yen"), ("the United Kingdom", "pound"),
        ("the United States", "dollar"), ("India", "rupee"),
        ("Russia", "ruble"), ("China", "yuan")]

REL_TMPL = {
    "cap":  ("The capital of {c} is {a}. ",  "The capital of {c} is"),
    "lang": ("People in {c} speak {a}. ",    "People in {c} speak"),
    "cur":  ("The currency of {c} is the {a}. ", "The currency of {c} is the"),
}
POOLS = {"cap": CAPS, "lang": LANGS, "cur": CURS}

# subjects + answers of EVERY candidate (kept or dropped): the wash corpus's
# banned strings (case-insensitive substring). The full phrases for the two
# multi-word subjects; generic template words (capital, currency, people,
# speak, located...) are NOT banned — e170's convention: plain corpus minus
# probe CONTENT, not minus English.
BANNED_EXTRA = ["united kingdom", "united states"]

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) -----------
DROP_FRAC = 0.5            # "recall drops < 50% of baseline"
HOLD_FRAC = 0.8            # "recall holds >= 80% of baseline"
BY_STEP = 10               # "...by <= 10 steps" (checkpoint grid {2, 10})
THROUGH_STEP = 200         # "...through 200 steps at the moderate lr"
PPLX_MAX_MULT = 2.0        # "healthy perplexity (within 2x its start)"

REGISTERED_PREDICTION = {
    "two_step_wash": "TWO-STEP-WASH fires if: recall drops < 50% of baseline "
        "by <= 10 steps at healthy perplexity (within 2x its start) — the "
        "no-basin physics replicates at 124M scale; the finding goes "
        "cross-scale.",
    "resistant": "RESISTANT fires if: recall holds >= 80% of baseline through "
        "200 steps at the moderate lr — pretrained facts are wash-resistant; "
        "the finding is bounded to the lab's small-net regime; W019's "
        "field-facing line dies its final death.",
    "texture": "TEXTURE in between (slow decay, lr-sensitivity) => TEXTURE "
        "with the curves.",
    "operationalizations": "R(c) = battery mean p(answer first token) over "
        "the KEPT probes (the p(Z)-battery analog; one absolute ruler); "
        "retention = R(c)/R(0); '<50% by <=10 steps' = retention < 0.5 at a "
        "measured checkpoint in {2,10} with bank ppl <= 2x start at that "
        "checkpoint; '>=80% through 200 at the moderate lr' = the 5e-5 arm "
        "retention >= 0.8 at EVERY checkpoint {2,10,50,200} with ppl health "
        "at every checkpoint; order TWO-STEP-WASH -> RESISTANT -> TEXTURE; "
        "probe gate = single-token answer AND [(top-1, p>=0.8) OR (top-5, "
        "p>=0.5)], cap 20 by baseline p, floor 10 (6-9 runs flagged); "
        "co-reports frac_top1/frac_top5/per-probe retention/ppl/in-batch CE.",
    "registration": "dispatched by the e182 mission text (the arc's one true "
        "remaining inverter); bars frozen VERBATIM in this docstring before "
        "compute. Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
device_events: list[dict] = []
deviations: list[str] = [
    "NEW ORGANISM (the point of the cell): HF GPT-2 124M replaces the lab's "
    "TinyGPT lineage — every lab instrument is either ported verbatim "
    "(optimizer, device policy, draw discipline) or replaced by its external "
    "analog (cloze p(answer) battery for the p(Z) battery); no lab checkpoint "
    "is touched.",
    "Size envelope: 124,439,808 params > the lab's 100M free tier, within "
    "common.py's 500M ceiling with the stated reason (external validity; the "
    "mission requires the field's own standard organism).",
    "THE FEW-SHOT INSTRUMENT: bare zero-shot cloze on GPT-2 124M peaks near "
    "p~0.2 on standard facts (session pilot: 1/36 candidates over the 0.8 "
    "bar), so the battery uses 2-shot primed prompts (GPT-2-era standard "
    "probing form). The fixed context is part of the frozen instrument; "
    "decay's LOCUS (fact vs in-context task-following) is not adjudicated "
    "here — flagged in the honesty reflex.",
    "Dropout disabled (all nn.Dropout p=0): the wash under test is the "
    "corpus-gradient pressure, not stochastic noise (e185 established noise "
    "kills; this cell isolates the stream). Side effect: bit-reproducible "
    "train-mode behavior.",
    "Wash corpus = the lab's data/input.txt (Shakespeare) LINE-FILTERED on "
    "every candidate subject+answer string (case-insensitive) + a token-id "
    "scan of the exact training stream for every kept answer token — stricter "
    "than the mission's zero-occurrence requirement where it is cheap "
    "(~620/40k lines dropped). Line-level filtering (vs e170's window-level "
    "rejection) because the stream here is whole-corpus, not a 16-window "
    "bank; the verification is the load-bearing part and it is exact.",
    "The 200-step wash covers ~3 epochs of the filtered corpus (batch 8 x 512 "
    "x 200 = 819k tokens over a ~300k-token corpus): windows resample with "
    "replacement, e176n's random-channel discipline.",
    "lr set {5e-6, 5e-5} is the dispatch's GPT-2-appropriate pair (the lab's "
    "1e-3/1e-4 axis does not transfer to a pretrained 124M); optimizer shape "
    "otherwise VERBATIM (AdamW (0.9,0.95) wd 0.1 constant lr clip 1.0).",
    "Final +200 checkpoints saved per arm to runs/checkpoints/e182_*.pt "
    "(gitignored; intermediates NOT saved — the recall battery is the record "
    "at this size).",
    "HF_HUB_OFFLINE=1 + revision pin: the organism is the local cache's "
    "607a30d snapshot exactly; no network dependency.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch); single commit, no push.",
    "Smoke mode trims: 4-step trainings, checkpoints {2,4}, 6-window bank, no "
    "cooldowns, nothing adjudicated.",
]


# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e184_seed_replicates.py VERBATIM (= e152r's pick_dev adapted
# to the lab's gpu_ok(); plus the dispatch's mid-run migration guard).

GPU_PARKED = False
PARK_REASON = None


def pick_dev(tag: str) -> torch.device:
    """Strict pre-training quick check (dispatch): gpu_ok() double-poll 5 s
    apart; PARK-ONCE — any failure parks every remaining training to CPU."""
    global GPU_PARKED, PARK_REASON
    if GPU_PARKED:
        log(f"[gpu] '{tag}' CPU (PARKED: {PARK_REASON})")
        return CPU
    if not torch.cuda.is_available():
        GPU_PARKED, PARK_REASON = True, "no CUDA"
        return CPU
    s1 = gpu_status()
    if gpu_ok():
        time.sleep(5)
        if gpu_ok():
            s2 = gpu_status()
            log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                f"{s2['mem_total']:.0f}MB)")
            return torch.device("cuda")
    GPU_PARKED = True
    PARK_REASON = f"quick check failed: {s1}"
    log(f"[gpu] PARK — '{tag}' and all remaining trainings run CPU ({s1})")
    return CPU


def migrate_to_cpu(net, opt) -> None:
    """Move net + optimizer state to CPU in place (params persist, so the
    opt.state keys stay valid). The dispatch's mid-run contention exit."""
    net.to("cpu")
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to("cpu")


# ------------------------------------------------------------------ organism

def load_organism():
    """Pinned pretrained GPT-2 124M from the local cache, dropout off."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    net = GPT2LMHeadModel.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    net.config.use_cache = False
    n_drop = 0
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            n_drop += 1
    n_params = sum(p.numel() for p in net.parameters())
    import transformers
    meta = {"repo": MODEL_REPO, "revision": MODEL_REV,
            "transformers_version": transformers.__version__,
            "params": int(n_params), "dropout_modules_zeroed": n_drop,
            "n_ctx": int(net.config.n_ctx), "n_layer": int(net.config.n_layer),
            "n_head": int(net.config.n_head), "n_embd": int(net.config.n_embd),
            "offline_cache": True,
            "size_reason": SIZE_REASON}
    log(f"organism: {MODEL_REPO}@{MODEL_REV[:8]} — {n_params:,} params, "
        f"{net.config.n_layer}L x {net.config.n_embd}d ctx {net.config.n_ctx}, "
        f"{n_drop} dropout modules zeroed")
    return tok, net, meta


# ------------------------------------------------------------------ probes

def build_candidates(tok) -> tuple[list[dict], list[dict]]:
    """Frozen candidate construction: 2-shot rotating prompts; single-token
    answer requirement. Returns (probed, dropped_multitoken)."""
    probed, dropped = [], []
    for rel, pool in POOLS.items():
        sent, query = REL_TMPL[rel]
        for i, (subj, ans) in enumerate(pool):
            ex = [pool[j] for j in range(len(pool)) if j != i][:K_SHOT]
            prefix = "".join(sent.format(c=c, a=a) for c, a in ex)
            prompt = prefix + query.format(c=subj)
            a_ids = tok.encode(" " + ans)
            rec = {"order": len(probed) + len(dropped), "relation": rel,
                   "subject": subj, "answer": ans, "fact": f"{subj}->{ans}",
                   "prompt": prompt,
                   "exemplars": [f"{c}->{a}" for c, a in ex],
                   "answer_ids": a_ids}
            if len(a_ids) != 1:
                rec["drop_reason"] = (f"answer tokenizes to {len(a_ids)} "
                                      f"tokens {a_ids} — first-token "
                                      f"measurement would not be exact")
                dropped.append(rec)
            else:
                rec["ans_id"] = a_ids[0]
                rec["ids"] = torch.tensor([tok.encode(prompt)], dtype=torch.long)
                probed.append(rec)
    return probed, dropped


@torch.no_grad()
def probe_one(net, ids: torch.Tensor, ans_id: int) -> dict:
    lg = net(input_ids=ids).logits[0, -1]
    p = F.softmax(lg, -1)
    pv = float(p[ans_id])
    rank = int((lg > lg[ans_id]).sum().item())
    return {"p": pv, "rank": rank, "top1": bool(rank == 0),
            "top5": bool(rank < 5)}


@torch.no_grad()
def probe_battery(net, battery: list[dict]) -> dict:
    """The recall battery on CPU: per-probe p(answer first token), rank;
    aggregates = the lab battery's mean-p(target) convention."""
    net.eval()
    rows = []
    for pr in battery:
        r = probe_one(net, pr["ids"], pr["ans_id"])
        rows.append({"fact": pr["fact"], "relation": pr["relation"], **r})
    ps = [r["p"] for r in rows]
    return {"probes": rows,
            "mean_p": float(sum(ps) / len(ps)),
            "frac_top1": float(sum(r["top1"] for r in rows) / len(rows)),
            "frac_top5": float(sum(r["top5"] for r in rows) / len(rows))}


def select_battery(cand_rows: list[dict]) -> tuple[list[str], dict]:
    """The frozen gate: keep iff (top-1 and p>=0.8) or (top-5 and p>=0.5);
    cap 20 by baseline p (ties by candidate order)."""
    for r in cand_rows:
        r["gate_pass"] = bool((r["top1"] and r["p"] >= GATE_TOP1_P)
                              or (r["top5"] and r["p"] >= GATE_TOP5_P))
    passers = [r for r in cand_rows if r["gate_pass"]]
    ranked = sorted(passers, key=lambda r: (-r["p"], r["order"]))
    kept_facts = [r["fact"] for r in ranked[:BATTRY_MAX]]
    for r in cand_rows:
        r["kept"] = bool(r["fact"] in kept_facts)
        if r["gate_pass"] and not r["kept"]:
            r["drop_reason"] = (f"cap {BATTRY_MAX}: baseline p below the "
                                f"kept cutoff")
    sel = {"gate": f"(top-1 and p>={GATE_TOP1_P}) or (top-5 and "
                   f"p>={GATE_TOP5_P}); cap {BATTRY_MAX} by baseline p, "
                   f"ties by candidate order",
           "n_candidates_probed": len(cand_rows),
           "n_pass": len(passers),
           "n_kept": len(kept_facts),
           "cutoff_p": (ranked[BATTRY_MAX - 1]["p"]
                        if len(ranked) >= BATTRY_MAX else None)}
    return kept_facts, sel


# ------------------------------------------------------------------ wash corpus

def build_wash_corpus(tok, text: str, banned: list[str],
                      answer_ids: dict[str, int]):
    """Line-filter the plain corpus on the banned strings; verify ZERO
    occurrences (string-level case-insensitive, then token-id level on the
    exact training stream); split a held-out tail bank."""
    lines = text.split("\n")
    kept_lines, dropped_lines = [], 0
    for ln in lines:
        low = ln.lower()
        if any(b in low for b in banned):
            dropped_lines += 1
        else:
            kept_lines.append(ln)
    filtered = "\n".join(kept_lines)          # original case preserved
    low = filtered.lower()

    before_counts = {b: text.lower().count(b) for b in banned}
    after_counts = {b: low.count(b) for b in banned}
    G_STR = {"rule": "drop any line containing a banned string "
                     "(case-insensitive substring)",
             "banned": banned,
             "lines_total": len(lines), "lines_dropped": dropped_lines,
             "occurrences_before": before_counts,
             "occurrences_after": after_counts,
             "max_after": max(after_counts.values())}
    G_STR["pass"] = bool(max(after_counts.values()) == 0)
    assert G_STR["pass"], f"string verification FAILED: {G_STR}"

    ids = torch.tensor(tok.encode(filtered), dtype=torch.long)
    tok_counts = {}
    for fact, aid in answer_ids.items():
        tok_counts[fact] = int((ids == aid).sum().item())
    G_TOK = {"rule": "count of each KEPT answer's first token id in the "
                     "exact training token stream",
             "counts": tok_counts, "max": max(tok_counts.values())}
    G_TOK["pass"] = bool(max(tok_counts.values()) == 0)
    assert G_TOK["pass"], f"token-id verification FAILED: {G_TOK}"

    n_bank = BANK_WINDOWS * SEQ
    train_ids = ids[: len(ids) - n_bank - 1]
    bank_span = ids[len(ids) - n_bank - 1:]
    bank_x = bank_span[:-1].view(BANK_WINDOWS, SEQ)
    bank_y = bank_span[1:].view(BANK_WINDOWS, SEQ)
    stats = {"source": "data/input.txt (the lab's plain corpus, Shakespeare)",
             "chars_before": len(text), "chars_after": len(filtered),
             "tokens_after": int(ids.shape[0]),
             "train_tokens": int(train_ids.shape[0]),
             "bank_windows": BANK_WINDOWS, "bank_seq": SEQ,
             "bank_is_held_out_tail": True,
             "epochs_over_200_steps": round(
                 BATCH * SEQ * 200 / max(int(train_ids.shape[0]), 1), 2)}
    return train_ids, (bank_x, bank_y), G_STR, G_TOK, stats


@torch.no_grad()
def ppl_eval(net, bank_x, bank_y, bs=4) -> dict:
    """Held-out wash-corpus CE -> perplexity (the health guard)."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, bank_x.shape[0], bs):
        logits = net(input_ids=bank_x[i:i + bs]).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               bank_y[i:i + bs].reshape(-1),
                               reduction="sum")
        tot += float(loss.item())
        n += bank_y[i:i + bs].numel()
    ce = tot / max(n, 1)
    return {"ce": ce, "ppl": math.exp(ce)}


# ------------------------------------------------------------------ the wash

def finetune_wash(tag: str, net0, train_ids: torch.Tensor, bank_xy,
                  battery: list[dict], lr: float,
                  ckpt_steps: tuple[int, ...]):
    """THE PLAIN-CORPUS WASH (e176n/e180's finetune_freeze arithmetic on the
    GPT-2 organism): AdamW (0.9,0.95) wd 0.1 constant lr, clip 1.0, full-token
    CE; per step draw BATCH window offsets from a CPU torch.Generator
    (device-independent); snapshots + CPU-side readouts (battery + bank ppl)
    at the checkpoint steps; time cap; mid-run GPU contention guard every 25
    steps (migrate to CPU and finish there)."""
    dev = pick_dev(tag)
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    hi = train_ids.shape[0] - SEQ - 1
    evl = copy.deepcopy(net0)                    # CPU eval twin
    evl.eval()
    traj, t_start = [], time.time()
    step, final_sd = 0, None
    for step in range(1, n_steps + 1):
        off = torch.randint(hi, (BATCH,), generator=gen)
        x = torch.stack([train_ids[o: o + SEQ] for o in off]).to(dev)
        y = torch.stack([train_ids[o + 1: o + 1 + SEQ] for o in off]).to(dev)
        logits = net(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in ckpt_set or step % 25 == 0:
            log(f"  [{tag}] s{step:4d} corpus CE {float(loss.item()):.4f} "
                f"({time.time() - t_start:.0f}s)")
        if step in ckpt_set:
            sd = {k: v.detach().to("cpu", torch.float32).clone()
                  for k, v in net.state_dict().items()}
            evl.load_state_dict(sd)
            bat = probe_battery(evl, battery)
            hp = ppl_eval(evl, *bank_xy)
            if step == n_steps:
                final_sd = sd
            traj.append({"step": step, "mean_p": bat["mean_p"],
                         "frac_top1": bat["frac_top1"],
                         "frac_top5": bat["frac_top5"],
                         "bank_ce": hp["ce"], "bank_ppl": hp["ppl"],
                         "in_batch_ce": float(loss.item()),
                         "probes": {r["fact"]: {"p": r["p"], "rank": r["rank"]}
                                    for r in bat["probes"]},
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} recall {bat['mean_p']:.4f} "
                f"top1 {bat['frac_top1']:.2f} | bank ppl {hp['ppl']:.2f} "
                f"(in-batch CE {float(loss.item()):.4f})")
            del sd
        if (time.time() - t_start) > TRAIN_CAP_S:
            log(f"  [{tag}] time cap {TRAIN_CAP_S:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append(
                    {"tag": tag, "step": step, "event": "MID-RUN MIGRATION",
                     "status": s,
                     "note": "contention guard fired (mem > 85% or temp > 80C)"
                             " — net + optimizer state moved to CPU; training"
                             " finishes on CPU (e184/e152 precedent)"})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> "
                    f"migrating to CPU")
                migrate_to_cpu(net, opt)
                dev = CPU
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"traj": traj, "final_sd": final_sd, "steps_ran": step,
            "lr": lr, "seed": FREEZE_SEED,
            "initial_device": "cpu" if GPU_PARKED else "cuda",
            "final_device": str(dev), "time_cap_s": TRAIN_CAP_S}


# ------------------------------------------------------------------ plot

def make_plot(rd, arms, base, verdict, clause, adjudication):
    """THE FIGURE: the recall-vs-steps wash curves with the perplexity
    overlay (the deliverable), retention vs the registered bars, the
    discrete accuracies, and the probe table + verdict panel."""
    steps_all = [0] + list(CK_MAIN)
    cols = {5e-6: "tab:blue", 5e-5: "tab:red"}
    lbls = {5e-6: "lr 5e-6 (gentle)", 5e-5: "lr 5e-5 (moderate)"}
    R0 = base["mean_p"]

    def seq(lr, key):
        return steps_all, [base[key]] + [t[key] for t in arms[LR_TAG[lr]]["traj"]]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) THE WASH: recall vs steps + perplexity overlay
    ax = axes[0, 0]
    for lr in LRS:
        xs, ys = seq(lr, "mean_p")
        ax.plot(xs, ys, "o-", ms=7, lw=2.2, color=cols[lr], label=lbls[lr])
    ax.axhline(R0, color="gray", ls=":", lw=1.2)
    ax.annotate(f"baseline R0 {R0:.3f}", (0, R0), textcoords="offset points",
                xytext=(6, 4), fontsize=7.5, color="gray")
    ax.axhline(DROP_FRAC * R0, color="tab:purple", ls="--", lw=1.1)
    ax.annotate(f"50% of baseline (TWO-STEP bar)", (0, DROP_FRAC * R0),
                textcoords="offset points", xytext=(6, -12), fontsize=7,
                color="tab:purple")
    ax.axhline(HOLD_FRAC * R0, color="seagreen", ls="--", lw=1.1)
    ax.annotate("80% of baseline (RESISTANT bar)", (0, HOLD_FRAC * R0),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
                color="seagreen")
    axr = ax.twinx()
    for lr in LRS:
        xs, ys = seq(lr, "bank_ppl")
        axr.plot(xs, ys, "s:", ms=5, lw=1.3, color=cols[lr], alpha=0.45)
    axr.set_ylabel("wash-corpus perplexity (dotted, right axis)", fontsize=8)
    axr.axhline(PPLX_MAX_MULT * base["bank_ppl"], color="k", ls="-.", lw=0.8,
                alpha=0.5)
    axr.annotate(f"health ceiling 2x start "
                 f"({PPLX_MAX_MULT * base['bank_ppl']:.1f})",
                 xy=(steps_all[-1], PPLX_MAX_MULT * base["bank_ppl"]),
                 xytext=(-4, 4), textcoords="offset points", ha="right",
                 fontsize=6.5, color="k", alpha=0.75)
    ax.set_xlabel("plain-corpus wash steps (batch 8 x ctx 512, AdamW wd 0.1)")
    ax.set_ylabel("RECALL R = battery mean p(answer first token)")
    ax.set_ylim(-0.03, min(1.05, R0 * 1.12))
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("THE GPT-2 WASH — recall vs steps (perplexity overlay) "
                 f"-> {verdict}", fontsize=10)

    # (0,1) retention vs the registered bars
    ax = axes[0, 1]
    for lr in LRS:
        xs, ys = steps_all, [1.0] + [t["mean_p"] / R0 for t in
                                     arms[LR_TAG[lr]]["traj"]]
        ax.plot(xs, ys, "o-", ms=7, lw=2.2, color=cols[lr], label=lbls[lr])
        for s, y in zip(xs, ys):
            if s > 0:
                ax.annotate(f"{y:.3f}", (s, y), textcoords="offset points",
                            xytext=(4, 6), fontsize=7, color=cols[lr])
    ax.axhline(DROP_FRAC, color="tab:purple", ls="--", lw=1.2,
               label="0.50 — TWO-STEP-WASH (< by step 10, healthy)")
    ax.axhline(HOLD_FRAC, color="seagreen", ls="--", lw=1.2,
               label="0.80 — RESISTANT (5e-5 through 200)")
    ax.set_xlabel("wash steps")
    ax.set_ylabel("retention R(c)/R(0)")
    ax.set_ylim(-0.03, 1.1)
    ax.legend(fontsize=7.4, loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("retention vs the registered bars (no shopping)", fontsize=9.5)

    # (1,0) discrete accuracies
    ax = axes[1, 0]
    for lr in LRS:
        for key, mrk, al in (("frac_top1", "o", 1.0), ("frac_top5", "^", 0.55)):
            xs, ys = seq(lr, key)
            ax.plot(xs, ys, marker=mrk, lw=1.8, ms=6, color=cols[lr],
                    alpha=al,
                    label=f"{lbls[lr]} {key.replace('frac_', '')}")
    ax.set_xlabel("wash steps")
    ax.set_ylabel("fraction of battery")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("discrete recall: argmax-correct and top-5 containment",
                 fontsize=9.5)

    # (1,1) probe table + verdict
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.98
    ax.text(0.02, y, "THE BATTERY (kept probes; p(answer) baseline -> +200 "
            "per arm):", fontsize=8.4, va="top", family="monospace",
            weight="bold")
    y -= 0.030
    hdr = (f"  {'fact':22s} {'rel':4s} {'R0':>6s} {'5e-6@200':>9s} "
           f"{'ret':>6s} {'5e-5@200':>9s} {'ret':>6s}")
    ax.text(0.02, y, hdr, fontsize=6.6, va="top", family="monospace")
    y -= 0.024
    kept = adjudication["battery_facts"]
    tr5e6 = {t["step"]: t for t in arms["lr5e6"]["traj"]}
    tr5e5 = {t["step"]: t for t in arms["lr5e5"]["traj"]}
    last = CK_MAIN[-1]
    for fact in kept:
        b = next(r for r in base["probes"] if r["fact"] == fact)
        p6 = tr5e6[last]["probes"][fact]["p"] if last in tr5e6 else float("nan")
        p5 = tr5e5[last]["probes"][fact]["p"] if last in tr5e5 else float("nan")
        ax.text(0.02, y,
                f"  {fact:22s} {b['relation']:4s} {b['p']:6.3f} "
                f"{p6:9.3f} {p6 / b['p']:6.2f} {p5:9.3f} {p5 / b['p']:6.2f}",
                fontsize=6.2, va="top", family="monospace")
        y -= 0.0215
    y -= 0.012
    ax.text(0.02, y, f"E182 VERDICT: {verdict}", fontsize=9.2, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.036
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top", family="monospace")
        y -= 0.024

    fig.suptitle("E182 — THE GPT-2 WASH: the external-validity fuse "
                 f"(GPT-2 124M, plain-corpus wash, battery n="
                 f"{len(kept)}) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "gpt2_wash.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e182_smoke" if SMOKE else "e182")
    log(f"E182 THE GPT-2 WASH (smoke={SMOKE}) -> {rd}")
    log(f"compute: GPU allowed (park-once policy + mid-run guard), cooldown "
        f"{COOLDOWN_S:.0f}s around each training, per-training cap "
        f"{TRAIN_CAP_S:.0f}s; gpu at start: {gpu_status()}")

    # ---------------- organism
    tok, net0, org_meta = load_organism()
    G_SIZE = {"params": org_meta["params"], "ceiling": SIZE_CEILING,
              "reason": SIZE_REASON,
              "pass": bool(org_meta["params"] <= SIZE_CEILING)}
    assert G_SIZE["pass"], f"size envelope exceeded: {G_SIZE}"

    # ---------------- (A) probe selection (frozen gate on the pristine model)
    cand, dropped_mt = build_candidates(tok)
    log(f"candidates: {len(cand)} probed + {len(dropped_mt)} dropped "
        f"pre-probe (multi-token answers)")
    base_cand = probe_battery(net0, cand)
    for r, b in zip(cand, base_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    kept_facts, sel = select_battery(cand)
    reduced = sel["n_kept"] < BATTRY_MIN
    G_PROBE = {**sel, "floor": BATTRY_FLOOR, "reduced_flag": reduced,
               "pass": bool(sel["n_kept"] >= BATTRY_FLOOR)}
    assert G_PROBE["pass"], f"battery below floor: {sel}"
    if reduced:
        log(f"WARNING: battery reduced ({sel['n_kept']} < {BATTRY_MIN}) — "
            f"running FLAGGED")
    battery = [r for r in cand if r["kept"]]
    log(f"battery: {sel['n_kept']} kept of {sel['n_candidates_probed']} "
        f"probed ({sel['n_pass']} passed the gate"
        + (f", cap {BATTRY_MAX} at p>={sel['cutoff_p']:.3f}"
           if sel["cutoff_p"] is not None else "") + ")")
    for r in battery:
        log(f"  kept {r['fact']:28s} p0={r['p']:.3f} rank={r['rank']}")

    # ---------------- (B) the wash corpus (filter + verify)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    banned = sorted({s.lower() for rel in POOLS for s, _ in POOLS[rel]}
                    | {a.lower() for rel in POOLS for _, a in POOLS[rel]}
                    | set(BANNED_EXTRA))
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, bank_xy, G_STR, G_TOK, corpus_stats = build_wash_corpus(
        tok, text, banned, answer_ids)
    log(f"G_STR: {G_STR['lines_dropped']}/{G_STR['lines_total']} lines "
        f"dropped, {corpus_stats['chars_after']} chars keep; zero banned "
        f"occurrences after: PASS")
    log(f"G_TOK: answer-token scan over the {corpus_stats['train_tokens']}-"
        f"token stream: zero: PASS")
    log(f"corpus: {corpus_stats}")
    log("gates: G_SIZE, G_PROBE, G_STR, G_TOK PASS"
        + (" (G_PROBE reduced)" if reduced else ""))

    # ---------------- step-0 readouts (the baseline record)
    base = probe_battery(net0, battery)
    base_hp = ppl_eval(net0, *bank_xy)
    base.update({"bank_ce": base_hp["ce"], "bank_ppl": base_hp["ppl"]})
    R0 = base["mean_p"]
    log(f"STEP-0: recall R0 {R0:.4f} (top1 {base['frac_top1']:.2f} top5 "
        f"{base['frac_top5']:.2f}) | bank CE {base_hp['ce']:.4f} ppl "
        f"{base_hp['ppl']:.2f}")

    # ---------------- (B) the two wash arms
    arms: dict = {}
    for lr in LRS:
        tag = LR_TAG[lr]
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag} — THE WASH at lr {lr}: {STEPS} steps, batch {BATCH} "
            f"x ctx {SEQ}, AdamW (0.9,0.95) wd 0.1 clip 1.0, seed "
            f"{FREEZE_SEED}, checkpoints +{list(CK_MAIN)}")
        arm = finetune_wash(tag, net0, train_ids, bank_xy, battery, lr,
                            CK_MAIN)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after {tag}")
            cooldown(COOLDOWN_S)
        for t in arm["traj"]:
            t["retention"] = t["mean_p"] / R0
            t["ppl_ratio"] = t["bank_ppl"] / base["bank_ppl"]
        arms[tag] = arm

        if not SMOKE and arm["final_sd"] is not None:
            ck = common.REPO / "runs" / "checkpoints" / f"e182_gpt2_{tag}.pt"
            torch.save({"model": arm["final_sd"],
                        "meta": {"experiment": "e182", "steps": STEPS,
                                 "lr": lr, "seed": FREEZE_SEED,
                                 "desc": f"openai-community/gpt2@{MODEL_REV} "
                                 f"+ {STEPS}-step plain-corpus wash "
                                 f"(filtered Shakespeare), batch {BATCH} x "
                                 f"{SEQ}, AdamW (0.9,0.95) wd 0.1 constant "
                                 f"clip 1.0",
                                 "base": MODEL_REPO, "revision": MODEL_REV}},
                       ck)
            arm["ckpt"] = str(ck.relative_to(common.REPO)).replace("\\", "/")
            log(f"[ckpt] saved {ck.name}")
        del arm["final_sd"]

    # ---------------- (C) adjudication (registered clauses; no shopping)
    def crossing(lr, horizon):
        for t in arms[LR_TAG[lr]]["traj"]:
            if t["step"] <= horizon and t["retention"] < DROP_FRAC:
                return t
        return None

    cross = {lr: crossing(lr, BY_STEP) for lr in LRS}
    two_step = any(c is not None and c["ppl_ratio"] <= PPLX_MAX_MULT
                   for c in cross.values())
    mod = arms["lr5e5"]["traj"]
    resistant = bool(mod) and all(
        t["retention"] >= HOLD_FRAC and t["ppl_ratio"] <= PPLX_MAX_MULT
        for t in mod) and mod[-1]["step"] >= THROUGH_STEP

    first_under = {}
    for lr in LRS:
        u = crossing(lr, CK_MAIN[-1])
        first_under[str(lr)] = None if u is None else u["step"]

    if two_step:
        verdict = "TWO-STEP-WASH"
        fired = [f"lr {lr}: retention {c['retention']:.3f} at +{c['step']} "
                 f"(ppl {c['ppl_ratio']:.2f}x start)"
                 for lr, c in cross.items() if c is not None]
        clause = ("the no-basin physics REPLICATES at 124M: recall fell "
                  f"under 50% of baseline by step {BY_STEP} at healthy "
                  f"perplexity ({'; '.join(fired)}) — the lab's finding goes "
                  "cross-scale onto the field's own organism.")
    elif resistant:
        verdict = "RESISTANT"
        r200 = mod[-1]
        clause = (f"pretrained facts are WASH-RESISTANT: at the moderate lr "
                  f"(5e-5) recall held >= 80% of baseline through +{r200['step']} "
                  f"(final retention {r200['retention']:.3f}, ppl "
                  f"{r200['ppl_ratio']:.2f}x start; gentle arm final "
                  f"{arms['lr5e6']['traj'][-1]['retention']:.3f}) — the "
                  "finding is bounded to the lab's small-net regime; W019's "
                  "field-facing line dies its final death.")
    else:
        verdict = "TEXTURE"
        g200 = arms["lr5e6"]["traj"][-1]
        r200 = mod[-1]
        clause = ("neither registered bar fired: slow decay / lr-sensitivity "
                  f"in between — moderate lr (5e-5) retention "
                  f"{r200['retention']:.3f} at +{r200['step']} (min "
                  f"{min(t['retention'] for t in mod):.3f}), gentle lr (5e-6) "
                  f"retention {g200['retention']:.3f} at +{g200['step']}; "
                  "first checkpoint under the 50% bar: "
                  + ", ".join(f"lr {k}: " + (f"+{v}" if v is not None
                                              else "none by +200")
                              for k, v in first_under.items())
                  + ". Full curves reported; no bar shopping.")
    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"

    adjudication = {
        "bars": {"TWO_STEP_WASH": two_step, "RESISTANT": resistant,
                 "verdict": verdict, "clause": clause,
                 "order": "TWO-STEP-WASH -> RESISTANT -> TEXTURE"},
        "recall_definition": "R(c) = battery mean p(answer first token); "
                             "retention = R(c)/R(0)",
        "baseline_R0": R0,
        "crossing_by_10": {str(lr): (None if c is None else
                                     {"step": c["step"],
                                      "retention": c["retention"],
                                      "ppl_ratio": c["ppl_ratio"]})
                           for lr, c in cross.items()},
        "first_under_50pct_any_horizon": first_under,
        "min_retention": {LR_TAG[lr]: min(t["retention"] for t in
                                          arms[LR_TAG[lr]]["traj"])
                          for lr in LRS},
        "final_retention": {LR_TAG[lr]: arms[LR_TAG[lr]]["traj"][-1][
            "retention"] for lr in LRS},
        "ppl_max_ratio": {LR_TAG[lr]: max(t["ppl_ratio"] for t in
                                          arms[LR_TAG[lr]]["traj"])
                          for lr in LRS},
        "healthy_all_checkpoints": {LR_TAG[lr]: all(
            t["ppl_ratio"] <= PPLX_MAX_MULT for t in arms[LR_TAG[lr]]["traj"])
            for lr in LRS},
        "battery_facts": [r["fact"] for r in battery],
        "steps_ran": {LR_TAG[lr]: arms[LR_TAG[lr]]["steps_ran"] for lr in LRS},
    }
    log("=" * 78)
    log(f"E182 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  R0 {R0:.4f}; final retention: 5e-6 "
        f"{arms['lr5e6']['traj'][-1]['retention']:.4f}, 5e-5 "
        f"{arms['lr5e5']['traj'][-1]['retention']:.4f}; first-under-50%: "
        f"{first_under}")
    log(f"  device events: {device_events or 'none'}")

    # ---------------- outputs
    cand_record = [{k: r.get(k) for k in
                    ("fact", "relation", "subject", "answer", "prompt",
                     "exemplars", "p", "rank", "top1", "top5", "gate_pass",
                     "kept", "drop_reason")}
                   for r in cand]
    mt_record = [{k: r.get(k) for k in
                  ("fact", "relation", "answer", "drop_reason")}
                 for r in dropped_mt]
    metrics = {
        "experiment": "e182_gpt2_wash",
        "date": common.now_iso(),
        "registration": ("dispatched by the e182 mission text; bars frozen "
                         "VERBATIM in the module docstring before compute; "
                         "adjudicated against exactly that — no bar shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the field's own organism (pretrained GPT-2 124M) "
                     "show the lab's no-basin wash physics (T114/T119), or "
                     "is a real model's factual recall RESISTANT to a "
                     "plain-corpus wash?"),
        "organism": org_meta,
        "size_gate": G_SIZE,
        "probe_selection": {
            "gate": sel, "reduced_flag": reduced,
            "battery_floor": BATTRY_FLOOR,
            "few_shot_instrument": (f"K={K_SHOT} rotating exemplars from the "
                                    "probe's own relation (leave-self-out, "
                                    "frozen candidate order); a probe's "
                                    "answer never appears in its own prompt"),
            "baseline_all_candidates": cand_record,
            "dropped_multitoken": mt_record,
            "kept_facts": [r["fact"] for r in battery],
        },
        "gates": {"G_PROBE": G_PROBE, "G_STR": G_STR, "G_TOK": G_TOK},
        "corpus": corpus_stats,
        "baseline": base,
        "arms": {
            LR_TAG[lr]: {
                "lr": lr, "seed": FREEZE_SEED,
                "desc": f"{STEPS}-step plain-corpus wash (filtered "
                        f"Shakespeare), batch {BATCH} x ctx {SEQ}, AdamW "
                        f"(0.9,0.95) wd 0.1 constant clip 1.0",
                "steps_ran": arms[LR_TAG[lr]]["steps_ran"],
                "initial_device": arms[LR_TAG[lr]]["initial_device"],
                "final_device": arms[LR_TAG[lr]]["final_device"],
                "time_cap_s": arms[LR_TAG[lr]]["time_cap_s"],
                "ckpt": arms[LR_TAG[lr]].get("ckpt"),
                "traj": arms[LR_TAG[lr]]["traj"],
            } for lr in LRS
        },
        "adjudication": adjudication,
        "honesty_reflex": {
            "probe_selection_bias": ("the battery is, by construction, the "
                                     "facts GPT-2 recalls BEST (gate p>=0.8 "
                                     "top-1 / >=0.5 top-5, cap-20 of the "
                                     "passers) — hyper-consolidated facts, "
                                     "the most-plausibly-basin-deep ones the "
                                     "organism owns; this biases the cell "
                                     "TOWARD RESISTANT and against "
                                     "TWO-STEP-WASH: a wash kill on THIS "
                                     "battery is strong, resistance on it is "
                                     "an upper bound on the population's "
                                     "resistance"),
            "few_shot_conflation": ("the 2-shot context is part of the frozen "
                                    "instrument; decay conflates fact "
                                    "storage with in-context task-following "
                                    "(bare zero-shot cloze was too weak to "
                                    "measure: 1/36 candidates over 0.8 in the "
                                    "session pilot); frac_top1 co-reported as "
                                    "the discrete check"),
            "corpus_composition": ("the wash stream is filtered Shakespeare — "
                                   "archaic, verse-heavy, IN WebText's "
                                   "distribution family but far from it in "
                                   "style; a WebText-like wash could behave "
                                   "differently; the filter dropped "
                                   f"{G_STR['lines_dropped']} lines to "
                                   "guarantee content-freedom, so the "
                                   'stream is also not "natural" Shakespeare'),
            "single_seed": ("one wash seed (18202), one trajectory per lr, "
                            "one organism instance — e152R/e184's seed "
                            "lottery precedent: wash timing is seed-textured; "
                            "the bars here are coarse enough (50%/80% "
                            "retention, order-of-magnitude steps) that the "
                            "verdict class is the claim, not the exact step"),
            "checkpoint_resolution": ("the {2,10,50,200} grid brackets "
                                      "crossings coarsely — a crossing "
                                      "'by +10' could occur anywhere in "
                                      "(2,10]; first-under-bar is reported "
                                      "as a step, not a time"),
            "epochs": ("200 steps cover ~3 epochs of the ~300k-token "
                       "filtered corpus (resampled windows); repetition is "
                       "part of this wash's pressure, unlike a fresh-data "
                       "fine-tune"),
            "dropout_off": ("the wash isolates corpus gradients (dropout "
                            "zeroed); the field's default fine-tune (dropout "
                            "0.1) adds the noise channel e185 showed is "
                            "lethal on its own"),
        },
        "compute": {
            "train_device_policy": ("GPU allowed (park-once quick check + "
                                    "mid-run guard every 25 steps); cooldown "
                                    f"{COOLDOWN_S:.0f}s around each training; "
                                    f"per-training cap {TRAIN_CAP_S:.0f}s; "
                                    "all readouts CPU-side"),
            "gpu_parked": GPU_PARKED, "park_reason": PARK_REASON,
            "device_events": device_events,
            "torch_threads": torch.get_num_threads(),
            "cells_devices": {LR_TAG[lr]: {
                "initial": arms[LR_TAG[lr]]["initial_device"],
                "final": arms[LR_TAG[lr]]["final_device"]} for lr in LRS},
        },
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"model": MODEL_REPO, "revision": MODEL_REV,
                   "seq": SEQ, "batch": BATCH, "steps": STEPS,
                   "ckpt_steps": list(CK_MAIN), "lrs": list(LRS),
                   "freeze_seed": FREEZE_SEED, "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", metrics)
    png = make_plot(rd, arms, base, verdict, clause, adjudication)
    log(f"outputs: {rd / 'metrics.json'}, {png}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
