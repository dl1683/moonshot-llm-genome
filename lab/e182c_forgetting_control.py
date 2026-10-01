"""E182C PHASE 1 — THE GPT-2 FORGETTING CONTROL (supervisor C12-3; the
carried objection to T123/e182's "surgical signature").

WHY: e182 (T123) found GPT-2 124M's facts erode under the lr 5e-5
plain-corpus wash while wash-corpus perplexity IMPROVES (71 -> 35) — the
"surgical signature". Check-in 12 item 3's carried objection: the trivial
expectation is GENERIC forgetting — fine-tuning at 5e-5 degrades ANY held-out
recall while in-domain perplexity improves. Until fact probes are compared
with MATCHED non-fact probes under the SAME states, T123's erosion cannot be
distinguished from ordinary forgetting. This cell is that control: a matched
held-out named-entity cloze battery read under e182's own wash, beside the
verbatim fact battery and the perplexity reference.

THE INVENTORY FINDING (Rule 12 check, done BEFORE compute; it reshapes the
mechanics, not the bars): **e182 SAVED NO WASH STATES.** runs/checkpoints/
contains zero e182*.pt; e182's metrics.json records ckpt=null for BOTH arms
(its design saved only +200 finals; both arms were time-capped at s108/s80
before any +200 checkpoint existed). The dispatch brief's premise ("eval-only
on e182's saved states") is therefore unsatisfiable as written. WHAT SURVIVED:
(a) the pristine organism (HF cache, revision-pinned); (b) e182's COMPLETE
frozen wash recipe — corpus filter (63 banned strings, verifiable to the
token count), draw discipline (CPU torch.Generator seeded 18202 ->
device-independent bit-identical window draws), optimizer (AdamW (0.9,0.95)
wd 0.1 constant lr 5e-5, clip 1.0, batch 8 x ctx 512); (c) e182's RECORDED
fact-battery + perplexity readings at +2/+10/+50 (runs/e182/metrics.json).

THE RECOVERY PATH (frozen here before compute): REPLAY e182's 5e-5 wash on
CPU from the pristine organism — same corpus (rebuilt and asserted EQUAL to
e182's recorded filtered-corpus stats), same seed 18202 (same draws), same
optimizer, same batch — and read BOTH batteries on the replayed states at
{0, +2, +10, +50, +80}. +80 is e182's own final trained step (its time cap);
+2/+10/+50 are e182's measured checkpoints and DOUBLE as the verification
gate: the replay is only admitted as "e182's states" if its fact-battery
readings agree with e182's recorded ones within the registered tolerances
below. This is state RECONSTRUCTION of e182's own frozen trajectory, not a
new arm; the one irreducible deviation is steps 1-25 (e182 trained them on
GPU before its mid-run CPU migration; the replay is CPU end-to-end), and the
verification gate MEASURES that deviation — discharging, as a side product,
the "CPU-vs-GPU wash numerics unquantified" item deferred to e182c by
check-in 12 item 5c. The replay also SAVES the per-state weights
(runs/checkpoints/e182c_s{N}.pt) — the discipline e182 lacked, so no future
e182c phase has to replay again.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE compute; no
bar shopping — adjudicate against exactly this):
  - FORGETTING-GENERIC: "controls erode within ~1.5x of the fact probes'
    relative decline at the deepest state — T123's erosion is generic
    forgetting at 124M; the GPT-2 clause scopes to 'ordinary forgetting with
    improving perplexity', NOT a no-basin signature"
  - FACT-SPECIFIC: "controls hold <= 20% while the fact erodes >= 50% —
    fact-specific; the surgical signature STRENGTHENS; phase 2 (fresh corpus
    draws, GPU) unlocks"
  - MIXED: "curves verbatim; the scope sentence carries numbers"

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do not
move the bars):
  * "relative decline at state s" = 1 - R_b(s)/R_b(0) per battery b (facts,
    controls), where R = battery mean p(answer first token) — e182's recall
    ruler, one absolute ruler per battery, retention normalized per battery.
  * "the deepest state" = the deepest PROBED state of the replay (intended
    +80; if the run is trimmed, the deepest reached, DISCLOSED). A
    co-adjudication at +50 (e182's own deepest MEASURED checkpoint, the
    T123-stamp comparison) is reported alongside; the PRIMARY verdict is the
    deepest probed state.
  * "controls erode within ~1.5x of the fact probes' relative decline" :=
    fact_decl/1.5 <= ctrl_decl <= 1.5*fact_decl, AND fact_decl > 0.02 (the
    fact must actually erode — T123's premise — else nothing is "generic").
  * "controls hold <= 20%" := ctrl_decl <= 0.20; "the fact erodes >= 50%" :=
    fact_decl >= 0.50.
  * Adjudication order: FACT-SPECIFIC -> FORGETTING-GENERIC -> MIXED; every
    boolean reported regardless.
  * ADJUDICATION IS GATED on the replay-verification gate G_REPLAY (below);
    if it fails, the verdict is VERIFICATION-FAILED, curves reported, no bar
    read.
  * THE CONTROL BATTERY (frozen pool, in-script): matched held-out
    named-entity cloze items GPT-2 plausibly knows, NOT in the wash corpus,
    NOT fact-related (zero overlap with e182's 63 banned fact strings; no
    capitals/languages/currencies of countries). Three relations (company-
    founders / company-products / mythology), same 2-shot rotating
    leave-self-out exemplar construction VERBATIM (in which pool[0]+pool[1]
    is the fixed exemplar pair for every later probe — e182's mechanics),
    same cloze form, same probe gate VERBATIM (single-token answer;
    (top-1 and p>=0.8) or (top-5 and p>=0.5); cap 20 by baseline p, ties by
    candidate order; floor 10, 6-9 runs FLAGGED reduced, <6 aborts). Count
    target 20 = the fact battery's size. Draw provenance: the pool is
    hand-curated and frozen verbatim below (no sampling; nothing random —
    the ONLY draw in the cell is the wash replay, which reuses e182's own
    seed 18202); candidates are DROPPED pre-probe by (a) wash-corpus
    contamination scan (subject or answer string absent from the FROZEN
    filtered corpus, case-insensitive; answer token id absent from the exact
    training stream — the corpus cannot be re-filtered without changing the
    wash, so absence is VERIFIED, not enforced) or (b) multi-token answers
    (e182's rule). All drops recorded with reasons. The pool's design used
    five pristine-model screening rounds (t=0 information ONLY, after the
    bars were frozen and before any wash compute) — see SCREENING PROVENANCE
    in the module body: person-surname answers fail GPT-2's full-name
    completion bias; land/superlatives fail under any uniform template;
    mythology scores p 0.02-0.14 (kept in-pool as the recorded scarcity);
    the p>=0.5 population at 124M concentrates in unique-anchor brand items.
  * THE NEARREL CO-REPORT (frozen, NOT the registered battery): the fact
    battery's own template ("The capital of {c} is {a}.") over DISJOINT
    entities (US-state capitals) — same gate, curve co-reported at every
    state. It reads whether the shared TEMPLATE erodes independently of the
    fact entities (the few-shot-locus check e182 could not make). The
    frozen bars adjudicate fact vs the REGISTERED control battery only.
  * (c) the improving reference: wash-corpus perplexity on e182's held-out
    tail bank (same 24x512 tail windows), read at every state.

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_STATES — the saved-states inventory + regeneration provenance (this
    docstring's inventory finding, recorded in metrics).
  * G_REPRO_CORPUS — the rebuilt filtered corpus must match e182's recorded
    stats EXACTLY: 40001 lines total, 664 dropped, 1093972 chars,
    331770 tokens, 319481 train tokens, bank 24x512, banned list identical.
  * G_BATT_FACT — the verbatim fact battery must reproduce e182's baseline:
    same 20 kept facts (set equality), per-probe |dp| <= 0.010, |dR0| <=
    0.005, bank ppl within 2% of 71.34.
  * G_CTRL — the control gate discipline above (floor/abort), contamination
    scans zero for every KEPT control, zero overlap with e182's banned
    strings.
  * G_REPLAY — at EVERY shared checkpoint {+2, +10, +50}: |replay mean_p -
    e182 recorded| <= 0.030; bank ppl ratio in [0.90, 1.10]; and shape:
    fact_decl(+50) >= 0.15 (e182 recorded 0.338). In-batch CE co-reported.
  * G_PPL — bank ppl read at every state (the health/improving reference).

WHAT THE CONTROL GUARANTEES (the honesty core): NOTHING — the openness is
the point. Controls holding does not prove the facts special (they may be
weaker-consolidated); controls eroding does not prove generic (shared few-
shot locus, shared prompt family). Either way the scope sentence tightens;
phase 2 (>=2 fresh corpus draws, GPU) is queued behind g2g and is what buys
generality. HONESTY STAMPS: phase-1; states REGENERATED on CPU fp32 (steps
1-25 originally GPU — the deviation measured by G_REPLAY); single organism,
single wash seed (e182's own 18202), single frozen corpus draw; the 2-shot
locus conflation inherited verbatim from e182's instrument (fact storage vs
in-context task-following not adjudicated); +80 never probed by e182 itself
(its log's last read is s75 CE); control battery n=1 hand-curated pool.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (124M fp32 inference + the 80-step
frozen replay, small batches; torch threads 8; walls logged per step). No
GPU claim, no new training arms (the replay reconstructs e182's own frozen
trajectory; "no training" in the brief meant no NEW washes — the inventory
finding made reconstruction the only path to "the same states", and it is
disclosed here before compute). No NOTES/THINKING/QUEUE/STATE edits
(dispatch). Progressive PARTIAL metrics writes + resumable state after every
checkpoint (the standing disruption rule).

RECOVERY NOTE (third dispatch): two prior e182c agents died in machine
disruptions PRE-ARTIFACT; no e182c script survived (verified: git history +
lab/). The frozen brief is all that survived; this script is the first
e182c artifact. Bars above are the brief's, verbatim.

PROVENANCE: the fact battery (pools, templates, 2-shot construction, probe,
gate), the corpus filter+verify, ppl_eval and the wash arithmetic are
lab/e182_gpt2_wash.py VERBATIM or near-verbatim (adaptations: CPU-only, no
cooldown/mid-run guard (no GPU), state saving, both batteries); e182's
recorded values are read from runs/e182/metrics.json at runtime (never
transcribed). Builds on: e182/T123 (the parent), T114/T119 (the no-basin
arc), SUPERVISOR check-in 12 item 3 (the control letter), R59-ideator
(e182c phase-1 promotion). NEW: the control battery, the matched-state
comparison, the state regeneration + verification discipline, the saved
per-state weights.

Run:  cd lab && python e182c_forgetting_control.py     (E182C_SMOKE=1 for
      the shakedown: 2 replay steps, own smoke dir, nothing adjudicated)
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

torch.set_num_threads(8)                              # e182 convention

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import now_iso, run_dir, save_json           # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E182C_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e182c_smoke" if SMOKE else "e182c"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- organism (e182 VERBATIM) --------------------------------------------------
MODEL_REPO = "openai-community/gpt2"
MODEL_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SIZE_CEILING = 500_000_000
SIZE_REASON = ("external validity (inherited from e182, its organism and its "
               "wash; 124,439,808 params > the 100M free tier, <= the 500M "
               "ceiling with e182's stated reason — the field's own standard "
               "organism; no smaller pretrained model answers this question)")

# ---- the frozen wash (e182's 5e-5 eroding arm, VERBATIM constants) -------------
LR = 5e-5                                  # THE eroding arm (dispatch: use it)
SEQ = 512
BATCH = 8
STEPS = 2 if SMOKE else 80                 # e182's lr5e5 arm true final step
CK_MAIN: tuple[int, ...] = (1, 2) if SMOKE else (2, 10, 50, 80)
FREEZE_SEED = 18202                        # e182's own wash seed (same draws)
BANK_WINDOWS = 2 if SMOKE else 24          # e182's held-out tail bank

E182_METRICS = common.REPO / "runs" / "e182" / "metrics.json"
E182_RUN_LOG = common.REPO / "runs" / "e182_run2.log"

# ---- the fact battery (e182 VERBATIM) ------------------------------------------
K_SHOT = 2
GATE_TOP1_P = 0.8
GATE_TOP5_P = 0.5
BATTRY_MIN, BATTRY_MAX = 10, 20
BATTRY_FLOOR = 6

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
BANNED_EXTRA = ["united kingdom", "united states"]

# ---- the control battery (frozen pool; THE new instrument) ---------------------
# SCREENING PROVENANCE (frozen record): the pool was designed through five
# pristine-model screening rounds BEFORE any wash compute — t=0 information
# only (the gate and bars were frozen first; no wash state was touched
# during design). Findings that shaped the pool (all recorded in metrics):
#   (1) person-surname answers (authors/inventors/discoverers) FAIL GPT-2's
#       full-name completion bias (" Charles Dickens" -> first token
#       " Charles" wins; best surname p ~ 0.25);
#   (2) land/superlative items (Everest 0.47, Fuji 0.38) fail under any
#       UNIFORM template, and several answers are corpus-contaminated
#       (thor/jupiter/mars/amazon/apple/mercury);
#   (3) mythology description->name items score p 0.02-0.14 (mass splits
#       across the pantheon) — KEPT IN-POOL as the recorded scarcity;
#   (4) the passing population at 124M concentrates in UNIQUE-ANCHOR
#       brand/product items ("the electric car company founded by Elon Musk
#       is called Tesla") — itself a frequency-law observation (Kandpal
#       2023 line): sharp cloze recall at this scale lives in the most
#       repeated, lowest-entropy mappings. The fact battery's
#       capitals/languages/currencies are exactly such a family.
# EXEMPLAR MECHANICS (matched to e182): in e182's first-two-others
# construction, every probe beyond the first two uses pool[0]+pool[1] as a
# FIXED exemplar pair; the two relations below each lead with their most
# canonical pair so the fixed exemplars are syntactically parallel to the
# queries. The construction rule is e182's, VERBATIM; only the pool order
# (part of the frozen instrument) was chosen, by pristine screening.
CTRL_TMPL = {
    "found": ("{c} is called {a}. ", "{c} is called"),
    "make":  ("{c} is called {a}. ", "{c} is called"),
    "myth":  ("{c} is called {a}. ", "{c} is called"),
}
CTRL_POOLS = {
    "found": [
        ("The software company founded by Bill Gates", "Microsoft"),
        ("The social network founded by Mark Zuckerberg", "Facebook"),
        ("The electric car company founded by Elon Musk", "Tesla"),
        ("The rocket company founded by Elon Musk", "SpaceX"),
        ("The search engine founded by Larry Page", "Google"),
        ("The social network founded by Jack Dorsey", "Twitter"),
        ("The shoe company founded by Phil Knight", "Nike"),
        ("The coffee chain founded in Seattle", "Starbucks"),
        ("The online encyclopedia that anyone can edit", "Wikipedia"),
    ],
    "make": [
        ("The gaming console made by Microsoft", "Xbox"),
        ("The web browser made by Google", "Chrome"),
        ("The phone made by Apple", "iPhone"),
        ("The tablet made by Apple", "iPad"),
        ("The email service made by Google", "Gmail"),
        ("The music store made by Apple", "iTunes"),
        ("The game console made by Sony", "PlayStation"),
        ("The console made by Nintendo", "Wii"),
    ],
    "myth": [
        ("The king of the gods in Greek mythology", "Zeus"),
        ("The goddess of wisdom", "Athena"),
        ("The messenger of the gods", "Hermes"),
        ("The goddess of the hunt", "Artemis"),
        ("The trickster god in Norse mythology", "Loki"),
        ("The home of the gods in Norse mythology", "Asgard"),
        ("The god of the underworld", "Hades"),   # corpus-contaminated: scan drops
        ("The god of thunder", "Thor"),           # corpus-contaminated: scan drops
    ],
}
# Expected from the final-order screening: found 5 pass (Facebook 0.74,
# Tesla 0.73, SpaceX 0.77, Google 0.83, Wikipedia 0.69), make 7 pass (Xbox
# 0.62, Chrome 0.70, iPhone 0.64, iPad 0.77, Gmail 0.75, iTunes 0.69,
# PlayStation 0.78), myth 0 pass (kept as the recorded scarcity) -> battery
# ~12, unflagged (floor 10); passers' baseline p 0.62-0.83 vs the fact
# battery's 0.568-0.953 — difficulty-matched.

# ---- the NEAR-RELATION CO-REPORT (not the registered battery) ------------------
# Same template as the FACT battery ("The capital of {c} is {a}.") over
# DISJOINT entities (US states, not countries): not in the wash corpus, not
# in e182's battery, no overlap with its 63 banned strings. Purpose: the
# registered controls (found/make/myth) share NO template with the facts,
# so their holding/eroding cannot separate "the facts eroded" from "the
# capital-template stopped being followed"; nearrel CAN (it shares the
# template). A CO-REPORT — the frozen bars adjudicate fact vs the
# registered control battery only.
NEAR_TMPL = ("The capital of {c} is {a}. ", "The capital of {c} is")
NEAR_POOL = [
    ("California", "Sacramento"), ("Colorado", "Denver"),
    ("Massachusetts", "Boston"), ("Texas", "Austin"),
    ("Georgia", "Atlanta"), ("Arizona", "Phoenix"),
    ("Illinois", "Springfield"), ("Washington", "Olympia"),
    ("Oregon", "Salem"), ("Ohio", "Columbus"), ("New York", "Albany"),
]
# Screening (same fixed-exemplar mechanics): passers Boston 0.61, Atlanta
# 0.64, Columbus 0.52; Denver 0.43 / Austin 0.40 near-miss; several answers
# corpus-contaminated (phoenix/olympia/salem) -> scan drops.

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS) ----------------
GEN_BAND = 1.5            # "within ~1.5x"
GEN_FLOOR_FACT_DECL = 0.02  # the fact must actually erode
HOLD_DECL = 0.20          # "controls hold <= 20%" decline
ERODE_DECL = 0.50         # "the fact erodes >= 50%"

# ---- registered verification tolerances (frozen) -------------------------------
TOL_BASE_PROBE_DP = 0.010   # per-probe baseline dp vs e182 record
TOL_BASE_R0_DP = 0.005      # battery mean baseline dp
TOL_BASE_PPL_REL = 0.02     # bank ppl rel dev vs e182 baseline 71.34
TOL_REPLAY_DP = 0.030       # |replay mean_p - e182 recorded| at shared ckpts
TOL_REPLAY_PPL_LO, TOL_REPLAY_PPL_HI = 0.90, 1.10
TOL_REPLAY_SHAPE_DECL50 = 0.15  # e182 fact decline at +50 was 0.338

REGISTERED_PREDICTION = {
    "forgetting_generic": "controls erode within ~1.5x of the fact probes' "
        "relative decline at the deepest state — T123's erosion is generic "
        "forgetting at 124M; the GPT-2 clause scopes to 'ordinary forgetting "
        "with improving perplexity', NOT a no-basin signature",
    "fact_specific": "controls hold <= 20% while the fact erodes >= 50% — "
        "fact-specific; the surgical signature STRENGTHENS; phase 2 (fresh "
        "corpus draws, GPU) unlocks",
    "mixed": "curves verbatim; the scope sentence carries numbers",
    "operationalizations": "decline_b(s) = 1 - R_b(s)/R_b(0); deepest state "
        "= deepest probed (intended +80, e182's own final step; co-"
        "adjudication at +50); within-1.5x := fact_decl/1.5 <= ctrl_decl <= "
        "1.5*fact_decl AND fact_decl > 0.02; hold := ctrl_decl <= 0.20; "
        "erode := fact_decl >= 0.50; order FACT-SPECIFIC -> FORGETTING-"
        "GENERIC -> MIXED; adjudication gated on G_REPLAY; control gate = "
        "e182's VERBATIM (cap 20, floor 10, abort 6); state grid "
        "{0,2,10,50,80}",
    "registration": "bars frozen VERBATIM from the third-dispatch brief "
        "(predecessors killed pre-artifact; the brief is the registration); "
        "adjudicate against exactly this; no bar shopping",
}

trims: list[str] = []
deviations: list[str] = [
    "STATES REGENERATED, NOT LOADED (the inventory finding, Rule 12): e182 "
    "saved no wash states (ckpt=null both arms; runs/checkpoints/ has no "
    "e182*.pt) — the brief's 'eval-only on saved states' was unsatisfiable. "
    "This cell REPLAYS e182's frozen 5e-5 wash on CPU (same corpus asserted "
    "equal, same seed 18202 -> bit-identical draws, same optimizer) and "
    "admits the states via the G_REPLAY verification gate against e182's "
    "recorded +2/+10/+50 readings. Steps 1-25 ran on GPU in e182; the replay "
    "is CPU end-to-end — the gate measures that deviation (check-in 12 item "
    "5c's deferred 'CPU/GPU migration numerics', discharged as a side "
    "product).",
    "Per-state weights ARE saved this time (runs/checkpoints/e182c_s{N}.pt) "
    "— the discipline e182 lacked; future phases eval-only on disk states.",
    "+80 (e182's true final step, its time cap) is probed here for the first "
    "time; e182's own measurements stop at +50. The +80 read is a state on "
    "the frozen trajectory, certified by the +2/+10/+50 agreement.",
    "The control corpus-freedom is VERIFIED post-hoc (subject/answer strings "
    "and answer token ids absent from the FROZEN filtered stream), not "
    "enforced by re-filtering — re-filtering would change the wash. Drop "
    "reasons recorded per candidate.",
    "Control pool designed via five pristine-model screening rounds (t=0 "
    "information only; bars already frozen; no wash state touched): the "
    "person-surname, land/superlative and mythology families fail the gate "
    "at 124M (full-name completion bias; no uniform template; p 0.02-0.14) "
    "— the recorded scarcity that itself mirrors Kandpal 2023's frequency "
    "law. The passing population (unique-anchor brand items) defines the "
    "registered battery; mythology stays in-pool as documented failures. A "
    "near-relation co-report (US-state capitals, the fact battery's own "
    "template, disjoint entities) reads the shared-template locus; it is "
    "NOT part of the frozen bars.",
    "CPU-only envelope incl. the 80-step replay (the brief's 'no training' "
    "meant no NEW washes; reconstruction of the frozen trajectory is the "
    "only path to 'the same states', disclosed before compute). Walls "
    "logged per step.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: 2 replay steps, grid {1,2}, own smoke dir, nothing "
    "adjudicated or verified.",
    "RECOVERY NOTE (third dispatch): two prior e182c agents died pre-artifact; "
    "no e182c script survived (git + lab/ verified). This is the first "
    "e182c artifact; bars are the frozen brief's, verbatim.",
]


# ------------------------------------------------------------------ organism

def load_organism():
    """Pinned pretrained GPT-2 124M from the local cache, dropout off
    (e182 VERBATIM)."""
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
            "offline_cache": True, "size_reason": SIZE_REASON,
            "dtype": "float32"}
    log(f"organism: {MODEL_REPO}@{MODEL_REV[:8]} — {n_params:,} params "
        f"(CPU fp32)")
    return tok, net, meta


# ------------------------------------------------------------------ probes
# build_candidates / probe_one / probe_battery / select_battery: e182 VERBATIM
# (select_battery additionally used for controls with their own rows).

def build_candidates(tok) -> tuple[list[dict], list[dict]]:
    """Frozen fact-battery candidate construction (e182 VERBATIM): 2-shot
    rotating prompts, single-token answer requirement."""
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
    """The recall battery on CPU: per-probe p(answer first token), rank
    (e182 VERBATIM; used for BOTH batteries)."""
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
    """The frozen gate (e182 VERBATIM): keep iff (top-1 and p>=0.8) or
    (top-5 and p>=0.5); cap 20 by baseline p (ties by candidate order)."""
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


# ------------------------------------------------------- the control battery

def build_control_candidates(tok, filtered_lower: str, train_ids,
                             e182_banned: list[str],
                             pools: dict, tmpls: dict) -> tuple[list[dict],
                                                               list[dict]]:
    """THE CONTROL POOLS (frozen): matched held-out named-entity cloze items.
    Serves the registered control battery (found/make/myth) and the nearrel
    co-report pool. Pre-probe drops, in order: (1) overlap with e182's
    banned fact strings ('fact-related'); (2) wash-corpus contamination —
    subject or answer string present in the FROZEN filtered corpus
    (case-insensitive), or answer token id present in the exact training
    stream; (3) multi-token answers (e182's rule). Survivors probed, then
    the VERBATIM gate."""
    probed, dropped = [], []
    banned_set = set(e182_banned)
    for rel, pool in pools.items():
        sent, query = tmpls[rel]
        for i, (subj, ans) in enumerate(pool):
            reason = None
            sl, al = subj.lower(), ans.lower()
            if sl in banned_set or al in banned_set:
                reason = (f"overlap with e182's banned fact strings "
                          f"({sl if sl in banned_set else al})")
            elif sl in filtered_lower:
                reason = f"subject string '{subj}' occurs in the wash corpus"
            elif al in filtered_lower:
                reason = f"answer string '{ans}' occurs in the wash corpus"
            a_ids = tok.encode(" " + ans)
            if reason is None and len(a_ids) != 1:
                reason = (f"answer tokenizes to {len(a_ids)} tokens "
                          f"{a_ids} — first-token measurement would not be "
                          f"exact")
            ex = [pool[j] for j in range(len(pool)) if j != i][:K_SHOT]
            prefix = "".join(sent.format(c=c, a=a) for c, a in ex)
            prompt = prefix + query.format(c=subj)
            rec = {"order": len(probed) + len(dropped), "relation": rel,
                   "subject": subj, "answer": ans,
                   "fact": f"{subj}->{ans}", "prompt": prompt,
                   "exemplars": [f"{c}->{a}" for c, a in ex],
                   "answer_ids": a_ids}
            if reason is not None:
                rec["drop_reason"] = reason
                dropped.append(rec)
            else:
                rec["ans_id"] = a_ids[0]
                rec["ids"] = torch.tensor([tok.encode(prompt)],
                                          dtype=torch.long)
                # token-level scan on the EXACT training stream
                n_tok = int((train_ids == rec["ans_id"]).sum().item())
                if n_tok > 0:
                    rec["drop_reason"] = (f"answer token id {rec['ans_id']} "
                                          f"occurs {n_tok}x in the exact "
                                          f"training stream")
                    dropped.append(rec)
                else:
                    rec["train_token_count"] = n_tok
                    probed.append(rec)
    return probed, dropped


# ------------------------------------------------------------------ corpus

def build_wash_corpus(tok, text: str, banned: list[str],
                      answer_ids: dict[str, int]):
    """e182's filter VERBATIM (fact strings only — the wash corpus is
    FROZEN; controls are verified against it, never re-filter it)."""
    lines = text.split("\n")
    kept_lines, dropped_lines = [], 0
    for ln in lines:
        low = ln.lower()
        if any(b in low for b in banned):
            dropped_lines += 1
        else:
            kept_lines.append(ln)
    filtered = "\n".join(kept_lines)
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

    ids = torch.tensor(tok.encode(filtered), dtype=torch.long)
    tok_counts = {}
    for fact, aid in answer_ids.items():
        tok_counts[fact] = int((ids == aid).sum().item())
    G_TOK = {"rule": "count of each KEPT answer's first token id in the "
                     "exact training token stream",
             "counts": tok_counts, "max": max(tok_counts.values())}
    G_TOK["pass"] = bool(max(tok_counts.values()) == 0)

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
             "bank_is_held_out_tail": True}
    return train_ids, (bank_x, bank_y), filtered, G_STR, G_TOK, stats


@torch.no_grad()
def ppl_eval(net, bank_x, bank_y, bs=4) -> dict:
    """Held-out wash-corpus CE -> perplexity (e182 VERBATIM)."""
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


# ------------------------------------------------------------------ the replay

def replay_wash(net0, train_ids, bank_xy, ckpt_steps: tuple[int, ...],
                on_ckpt, resume=None):
    """THE 5e-5 WASH REPLAY (e182's finetune_wash adapted): CPU-only, AdamW
    (0.9,0.95) wd 0.1 constant lr 5e-5, clip 1.0, full-token CE; per step
    draw BATCH window offsets from the CPU generator seeded 18202 (e182's
    device-independent discipline — bit-identical draws). At each checkpoint
    step: call on_ckpt(step, state_dict, in_batch_ce) and save the resumable
    state (model+opt+generator) + the per-state weights archive."""
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    hi = train_ids.shape[0] - SEQ - 1
    step = 0
    ckpt_dir = common.REPO / "runs" / "checkpoints"
    latest = ckpt_dir / (f"{NAME}_replay_latest.pt")
    if resume is not None:
        net.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        gen.set_state(resume["gen"])
        step = int(resume["step"])
        log(f"replay: RESUMED from step {step} (draw sequence continues "
            f"bit-identically from the saved generator state)")
    ckpt_set = set(ckpt_steps)
    t_start = time.time()
    t_mark = t_start
    n_steps = ckpt_steps[-1]
    while step < n_steps:
        step += 1
        off = torch.randint(hi, (BATCH,), generator=gen)
        x = torch.stack([train_ids[o: o + SEQ] for o in off])
        y = torch.stack([train_ids[o + 1: o + 1 + SEQ] for o in off])
        logits = net(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step in ckpt_set:
            dt = time.time() - t_mark
            t_mark = time.time()
            log(f"  [replay] s{step:3d}/{n_steps} corpus CE "
                f"{float(loss.item()):.4f} ({dt:.1f}s since last)")
        if step in ckpt_set:
            sd = {k: v.detach().to("cpu", torch.float32).clone()
                  for k, v in net.state_dict().items()}
            on_ckpt(step, sd, float(loss.item()))
            torch.save({"model": sd,
                        "meta": {"experiment": NAME, "step": step, "lr": LR,
                                 "seed": FREEZE_SEED,
                                 "desc": f"openai-community/gpt2@{MODEL_REV} "
                                 f"e182 5e-5 wash REPLAY (CPU fp32), step "
                                 f"{step}", "base": MODEL_REPO,
                                 "revision": MODEL_REV}},
                       ckpt_dir / f"{NAME}_s{step}.pt")
            torch.save({"model": sd,
                        "opt": opt.state_dict(),
                        "gen": gen.get_state(),
                        "step": step}, latest)
            del sd
    return step


# ------------------------------------------------------------------ plot

def make_plot(rd, states, verdict, clause, adj, form_match):
    """THE FIGURE: both batteries under the SAME states (mean_p + ppl
    overlay), per-battery retention vs the bar bands, discrete accuracies,
    and the verdict + verification + control-table panel."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green"}
    lbls = {"fact": "FACT battery (e182 verbatim, n=%d)" % len(adj["fact_facts"]),
            "ctrl": "CONTROL battery (matched, n=%d)" % len(adj["ctrl_facts"]),
            "near": "NEARREL co-report (same template, n=%d)"
                    % len(adj.get("nearrel_facts", []))}
    steps = [s["step"] for s in states]
    last = steps[-1]
    has_near = states[0].get("near") is not None

    def seq(b, key):
        if b == "near" and not has_near:
            return [], []
        return steps, [s[b][key] if s.get(b) else float("nan")
                       for s in states]

    R0 = {"fact": states[0]["fact"]["mean_p"],
          "ctrl": states[0]["ctrl"]["mean_p"],
          "near": states[0]["near"]["mean_p"] if has_near else None}

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) both batteries + perplexity overlay
    ax = axes[0, 0]
    for b in (("fact", "ctrl", "near") if has_near else ("fact", "ctrl")):
        xs, ys = seq(b, "mean_p")
        ax.plot(xs, ys, "o-", ms=7, lw=2.2, color=cols[b], label=lbls[b],
                alpha=0.55 if b == "near" else 1.0)
        if b != "near":
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(3, 6), fontsize=6.8, color=cols[b])
    axr = ax.twinx()
    xs, ys = steps, [s["ppl"] for s in states]
    axr.plot(xs, ys, "s:", ms=5, lw=1.4, color="seagreen", alpha=0.7)
    axr.set_ylabel("wash-corpus perplexity (dotted, right; IMPROVES)",
                   fontsize=8, color="seagreen")
    ax.set_xlabel("wash steps (e182's frozen 5e-5 arm, REPLAYED on CPU)")
    ax.set_ylabel("R = battery mean p(answer first token)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    ax.set_title("THE FORGETTING CONTROL — fact vs matched control probes "
                 f"under the SAME states -> {verdict}", fontsize=10)

    # (0,1) per-battery retention + bar bands
    ax = axes[0, 1]
    for b in (("fact", "ctrl", "near") if has_near else ("fact", "ctrl")):
        xs, ys = seq(b, "mean_p")
        ax.plot(xs, [y / R0[b] for y in ys], "o-", ms=7, lw=2.2,
                color=cols[b], label=lbls[b],
                alpha=0.55 if b == "near" else 1.0)
    ax.axhline(1 - HOLD_DECL, color="tab:blue", ls="--", lw=1.2,
               label="controls hold >= 80% (decline <= 20%)")
    ax.axhline(1 - ERODE_DECL, color="tab:red", ls="--", lw=1.2,
               label="fact erodes >= 50% (retention <= 0.50)")
    ax.set_xlabel("wash steps")
    ax.set_ylabel("retention R(s)/R(0) per battery")
    ax.set_ylim(-0.03, 1.12)
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25)
    _ratio = adj["declines_at_deepest"]["decline_ratio_ctrl_over_fact"]
    _ratio_txt = f"{_ratio:.2f}" if _ratio is not None else "n/a"
    ax.set_title(f"retention vs the frozen bars (deepest state +{last}; "
                 f"decline ratio ctrl/fact {_ratio_txt}, band "
                 f"[{1/GEN_BAND:.2f}, {GEN_BAND:.2f}])", fontsize=9.5)

    # (1,0) discrete accuracies + e182 overlay
    ax = axes[1, 0]
    for b in (("fact", "ctrl", "near") if has_near else ("fact", "ctrl")):
        xs, ys = seq(b, "frac_top1")
        ax.plot(xs, ys, "o-", ms=6, lw=1.8, color=cols[b],
                alpha=0.55 if b == "near" else 1.0,
                label=f"{b}: frac argmax-correct")
        xs, ys = seq(b, "frac_top5")
        ax.plot(xs, ys, "^:", ms=5, lw=1.4, color=cols[b], alpha=0.35,
                label=f"{b}: frac top-5")
    if adj.get("e182_fact_mean_p"):
        exs, eys = zip(*sorted((int(k) + 0.0, v) for k, v in
                               adj["e182_fact_mean_p"].items()))
        ax.plot(exs, eys, "x--", ms=9, lw=1.2, color="k", alpha=0.65,
                label="e182 RECORDED fact mean_p (the record)")
    ax.set_xlabel("wash steps")
    ax.set_ylabel("fraction of battery")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=6.8, loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("discrete recall + the e182 record overlay "
                 "(the verification spine)", fontsize=9.5)

    # (1,1) verdict + gates + control table
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.98
    ax.text(0.02, y, "VERIFICATION (the replay admits itself as e182's "
            "states):", fontsize=8.6, va="top", family="monospace",
            weight="bold")
    y -= 0.028
    for gname, gval in adj["gates_summary"].items():
        ax.text(0.02, y, f"  {gname:16s} {'PASS' if gval else 'FAIL'}",
                fontsize=7.2, va="top", family="monospace")
        y -= 0.021
    y -= 0.010
    ax.text(0.02, y, "CONTROL BATTERY (p0 -> deepest, retention):",
            fontsize=8.6, va="top", family="monospace", weight="bold")
    y -= 0.026
    hdr = (f"  {'control item':34s} {'rel':6s} {'p0':>6s} "
           f"{f'+{last}':>7s} {'ret':>6s}")
    ax.text(0.02, y, hdr, fontsize=6.6, va="top", family="monospace")
    y -= 0.022
    c0 = {r["fact"]: r["p"] for r in states[0]["ctrl"]["probes"]}
    cl = {r["fact"]: r["p"] for r in states[-1]["ctrl"]["probes"]}
    c_rel = {r["fact"]: r["relation"] for r in states[0]["ctrl"]["probes"]}
    for fact in adj["ctrl_facts"]:
        ax.text(0.02, y,
                f"  {fact:34s} {c_rel[fact]:6s} "
                f"{c0[fact]:6.3f} {cl[fact]:7.3f} "
                f"{cl[fact] / c0[fact]:6.2f}",
                fontsize=6.1, va="top", family="monospace")
        y -= 0.0195
        if y < 0.42:
            break
    y = 0.40
    ax.text(0.02, y, f"E182C VERDICT: {verdict}", fontsize=9.4, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.034
    for wd in textwrap.wrap(clause, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top", family="monospace")
        y -= 0.022
    y -= 0.008
    for wd in textwrap.wrap(form_match, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.2, va="top", family="monospace",
                color="dimgray")
        y -= 0.020

    fig.suptitle("E182C PHASE 1 — THE GPT-2 FORGETTING CONTROL (e182's "
                 f"5e-5 wash replayed CPU fp32, states +{steps}; fact n="
                 f"{len(adj['fact_facts'])} vs control n="
                 f"{len(adj['ctrl_facts'])}) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "forgetting_control.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    journal_path = rd / "journal.json"
    latest_ck = common.REPO / "runs" / "checkpoints" / f"{NAME}_replay_latest.pt"
    log(f"E182C PHASE 1 THE FORGETTING CONTROL (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e182c_forgetting_control",
        "phase": "1 (matched batteries under e182's frozen wash; phase 2 = "
                 "fresh corpus draws, queued for GPU behind g2g)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": ("bars frozen VERBATIM from the third-dispatch brief "
                         "(predecessors killed pre-artifact); adjudicated "
                         "against exactly that — no bar shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is e182/T123's fact erosion under the 5e-5 wash "
                     "GENERIC forgetting (matched controls erode with it) or "
                     "FACT-SPECIFIC (controls hold while facts erode)?"),
        "builds_on": ["e182/T123 (parent: the surgical signature)",
                      "SUPERVISOR check-in 12 item 3 (the control letter)",
                      "R59-ideator (e182c phase-1 promotion)",
                      "e176n/e180 (the wash arithmetic e182 inherited)"],
        "whats_new": ["the matched control battery (inventors/authors/"
                      "discoveries cloze, e182's gate verbatim)",
                      "the same-state comparison (fact vs control under one "
                      "wash trajectory)",
                      "the state-regeneration + verification discipline "
                      "(e182 saved no states)",
                      "per-state weights saved for future eval-only phases"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------------------- P0 record
    if not E182_METRICS.exists():
        log(f"FATAL: e182 record missing: {E182_METRICS}")
        return 1
    e182 = json.loads(E182_METRICS.read_text(encoding="utf-8"))
    e_arm = e182["arms"]["lr5e5"]
    e_base = e182["baseline"]
    e_traj = {t["step"]: t for t in e_arm["traj"]}
    e_str = e182["gates"]["G_STR"]
    e_corp = e182["corpus"]
    shared_steps = sorted(set(e_traj) & set(CK_MAIN))
    e_banned = e_str["banned"]

    # THE INVENTORY (Rule 12, done pre-compute; reshapes mechanics not bars)
    import glob as _glob
    ck_glob = list(_glob.glob(str(common.REPO / "runs" / "checkpoints"
                                   / "e182*.pt")))
    inv = {
        "checked": ["runs/e182*", "runs/checkpoints/e182*",
                    "runs/checkpoints/*gpt2*"],
        "e182_ckpt_files_found": ck_glob,
        "e182_metrics_ckpt_fields": {
            tag: e182["arms"][tag].get("ckpt") for tag in e182["arms"]},
        "e182_steps_ran": {tag: e182["arms"][tag]["steps_ran"]
                           for tag in e182["arms"]},
        "finding": ("NO e182 wash states exist: e182's design saved only "
                    "+200 finals; both arms were time-capped (lr5e6 s108, "
                    "lr5e5 s80) before step 200, so ckpt=null; zero "
                    "e182*.pt on disk. The brief's 'eval-only on saved "
                    "states' was unsatisfiable; this cell REPLAYS the frozen "
                    "5e-5 wash on CPU (corpus asserted equal, seed 18202 "
                    "bit-identical draws, same optimizer) and admits the "
                    "states via G_REPLAY vs e182's recorded +2/+10/+50 "
                    "readings."),
        "what_survived": ["the pristine organism (HF cache, pinned)",
                          "the frozen wash recipe (corpus filter + seed "
                          "18202 + optimizer)", "e182's recorded readings "
                          "(runs/e182/metrics.json)"],
        "regeneration_disclosed_before_compute": True,
    }
    G_STATES = {**inv, "pass": True}
    metrics["inventory"] = inv
    log(f"G_STATES: {inv['finding']}")
    log(f"e182 record: lr5e5 steps_ran={inv['e182_steps_ran']['lr5e5']}, "
        f"measured ckpts {sorted(e_traj)}, R0 "
        f"{e_base['mean_p']:.4f}, bank ppl {e_base['bank_ppl']:.2f}")

    # ---------------------------------------------------------------- organism
    tok, net0, org_meta = load_organism()
    metrics["organism"] = org_meta
    G_SIZE = {"params": org_meta["params"], "ceiling": SIZE_CEILING,
              "reason": SIZE_REASON,
              "pass": bool(org_meta["params"] <= SIZE_CEILING)}
    assert G_SIZE["pass"], f"size envelope exceeded: {G_SIZE}"
    metrics["size_gate"] = G_SIZE

    # ------------------------------------------- P1a the FACT battery (verbatim)
    cand, dropped_mt = build_candidates(tok)
    base_cand = probe_battery(net0, cand)
    for r, b in zip(cand, base_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    kept_facts, sel = select_battery(cand)
    battery = [r for r in cand if r["kept"]]

    e_kept = e182["adjudication"]["battery_facts"]
    e_base_probes = {r["fact"]: r["p"] for r in e_base["probes"]}
    dp = {r["fact"]: abs(r["p"] - e_base_probes[r["fact"]])
          for r in base_cand["probes"] if r["fact"] in e_base_probes}
    G_BATT_FACT = {
        "kept_set_equal": bool(set(kept_facts) == set(e_kept)),
        "max_per_probe_dp": max(dp.values()) if dp else None,
        "mean_per_probe_dp": (sum(dp.values()) / len(dp)) if dp else None,
        "tol_per_probe_dp": TOL_BASE_PROBE_DP,
        "R0_e182": e_base["mean_p"],
        "tol_R0_dp": TOL_BASE_R0_DP,
        "n_kept": len(kept_facts),
        "note": "R0 comparison happens at t=0 (P2) against the KEPT-20 "
                "battery mean (e182's baseline mean is over its battery, "
                "not its candidate pool)",
    }
    G_BATT_FACT["pass"] = bool(
        G_BATT_FACT["kept_set_equal"]
        and max(dp.values()) <= TOL_BASE_PROBE_DP)
    metrics["fact_battery"] = {
        "source": "e182's battery VERBATIM (pools/templates/2-shot/gate)",
        "gate": sel, "kept_facts": kept_facts,
        "baseline_all_candidates": [
            {k: r.get(k) for k in ("fact", "relation", "subject", "answer",
                                   "prompt", "p", "rank", "top1", "top5",
                                   "gate_pass", "kept", "drop_reason")}
            for r in cand],
        "dropped_multitoken": [{k: r.get(k) for k in ("fact", "relation",
                                                      "answer",
                                                      "drop_reason")}
                               for r in dropped_mt],
        "verification_vs_e182": G_BATT_FACT,
    }
    log(f"fact battery: {len(kept_facts)} kept (e182 record: {len(e_kept)}); "
        f"max per-probe dp {max(dp.values()):.6f} "
        f"{'PASS' if G_BATT_FACT['pass'] else 'FAIL'} "
        f"(R0 comparison deferred to t=0)")
    assert set(kept_facts) == set(e_kept), "fact battery diverged from e182"

    # ------------------------------------- P1b the frozen corpus (rebuild+assert)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    banned = sorted({s.lower() for rel in POOLS for s, _ in POOLS[rel]}
                    | {a.lower() for rel in POOLS for _, a in POOLS[rel]}
                    | set(BANNED_EXTRA))
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, bank_xy, filtered, G_STR, G_TOK, corpus_stats = (
        build_wash_corpus(tok, text, banned, answer_ids))
    filter_equal = bool(
        banned == e_banned
        and G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"])
    bank_equal = bool(
        corpus_stats["train_tokens"] == e_corp["train_tokens"]
        and corpus_stats["bank_windows"] == e_corp["bank_windows"])
    G_REPRO_CORPUS = {
        "banned_list_identical": bool(banned == e_banned),
        "lines_total": [G_STR["lines_total"], e_str["lines_total"]],
        "lines_dropped": [G_STR["lines_dropped"], e_str["lines_dropped"]],
        "chars_after": [corpus_stats["chars_after"], e_corp["chars_after"]],
        "tokens_after": [corpus_stats["tokens_after"], e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "bank_windows": [corpus_stats["bank_windows"], e_corp["bank_windows"]],
        "format": "[replayed, e182_recorded]",
        "filter_equal": filter_equal,
        "bank_equal": bank_equal,
        "pass": bool(filter_equal and (bank_equal or SMOKE)),
        "note": "smoke trims the bank (2 windows) so train_tokens differs "
                "by design there; the FILTER equality (lines/chars/tokens "
                "of the filtered text) is the load-bearing check and must "
                "hold in every mode",
    }
    log(f"G_REPRO_CORPUS: {'PASS' if G_REPRO_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens, "
        f"{G_STR['lines_dropped']} lines dropped; e182 "
        f"{e_corp['tokens_after']}/{e_str['lines_dropped']})")
    assert G_REPRO_CORPUS["pass"], f"corpus rebuild diverged: {G_REPRO_CORPUS}"

    # ------------------------------------------- P1c the CONTROL battery (new)
    ccand, cdropped = build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned, CTRL_POOLS, CTRL_TMPL)
    cbase_cand = probe_battery(net0, ccand)
    for r, b in zip(ccand, cbase_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    ctrl_kept, csel = select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]
    reduced = csel["n_kept"] < BATTRY_MIN
    G_CTRL = {**csel, "floor": BATTRY_FLOOR, "reduced_flag": reduced,
              "pool": {rel: len(pool) for rel, pool in CTRL_POOLS.items()},
              "drops": [{"fact": r["fact"], "reason": r["drop_reason"]}
                        for r in cdropped],
              "kept_contamination_scans_zero": True,
              "kept_overlap_with_e182_banned_zero": True,
              "pass": bool(csel["n_kept"] >= BATTRY_FLOOR)}
    metrics["control_battery"] = {
        "definition": ("matched held-out named-entity cloze: three "
                       "relations (company-founders / company-products / "
                       "mythology — the last kept as the recorded scarcity: "
                       "its p 0.02-0.14 members document that 124M's p>=0.5 "
                       "cloze population is thin outside unique-anchor "
                       "items), e182's 2-shot construction and gate "
                       "VERBATIM, count target 20"),
        "templates": {k: list(v) for k, v in CTRL_TMPL.items()},
        "pool_frozen_in_script": True,
        "draw_provenance": ("no sampling anywhere: the pool is frozen "
                            "verbatim; exemplars are the first two "
                            "leave-self-out candidates in frozen order "
                            "(e182's rule, so pool[0]+pool[1] is the fixed "
                            "exemplar pair for the relation); the ONLY "
                            "random element in the cell is the wash "
                            "replay's window draws, which reuse e182's own "
                            "seed 18202"),
        "screening_provenance": ("five pristine-model screening rounds, t=0 "
                                 "information only, AFTER the bars were "
                                 "frozen and BEFORE any wash compute; see "
                                 "the module docstring's SCREENING "
                                 "PROVENANCE"),
        "gate": {k: v for k, v in G_CTRL.items() if k != "drops"},
        "kept_facts": ctrl_kept,
        "baseline_all_candidates": [
            {k: r.get(k) for k in ("fact", "relation", "subject", "answer",
                                   "prompt", "p", "rank", "top1", "top5",
                                   "gate_pass", "kept")}
            for r in ccand],
        "dropped_pre_probe": G_CTRL["drops"],
    }
    cmsg = (f"control battery: {csel['n_kept']} kept of "
            f"{csel['n_candidates_probed']} probed ({csel['n_pass']} passed"
            + (f", cap {BATTRY_MAX} at p>={csel['cutoff_p']:.3f}"
               if csel["cutoff_p"] is not None else "") + "); "
            + (f"FLAGGED reduced (<{BATTRY_MIN})" if reduced else
               "count ok"))
    log(cmsg)
    if not SMOKE:
        assert G_CTRL["pass"], f"control battery below floor: {csel}"

    # --------------------------- P1d the NEARREL co-report battery (same gate)
    ncand, ndropped = build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": NEAR_POOL}, {"near": NEAR_TMPL})
    nbase_cand = probe_battery(net0, ncand)
    for r, b in zip(ncand, nbase_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    near_kept, nsel = select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]
    metrics["nearrel_battery"] = {
        "definition": ("CO-REPORT (not the registered battery; the frozen "
                       "bars adjudicate fact vs the registered control "
                       "battery only): SAME template as the fact battery "
                       "('The capital of {c} is {a}.') over DISJOINT "
                       "entities (US states). Reads whether a shared "
                       "template erodes independently of the facts — the "
                       "few-shot-locus check e182 could not make"),
        "template": list(NEAR_TMPL),
        "gate": {**nsel, "floor": None, "note": "no floor/abort: a "
                 "co-report; curve reported for whatever passes"},
        "kept_facts": near_kept,
        "baseline_all_candidates": [
            {k: r.get(k) for k in ("fact", "relation", "subject", "answer",
                                   "p", "rank", "top1", "top5", "gate_pass",
                                   "kept")}
            for r in ncand],
        "dropped_pre_probe": [{"fact": r["fact"],
                               "reason": r["drop_reason"]}
                              for r in ndropped],
    }
    log(f"nearrel co-report: {nsel['n_kept']} kept of "
        f"{nsel['n_candidates_probed']} probed "
        f"({nsel['n_pass']} passed the gate)")

    # form-matching record (Rule 12 check)
    f_p0 = [r["p"] for r in cand if r["kept"]]
    c_p0 = [r["p"] for r in ccand if r["kept"]]
    n_p0 = [r["p"] for r in ncand if r["kept"]]
    form_match = (
        "FORM-MATCHING: all batteries 2-shot rotating leave-self-out cloze, "
        f"single-token answers, gate (top1 p>=0.8)|(top5 p>=0.5), cap 20; "
        f"fact n={len(f_p0)} (p0 {min(f_p0):.3f}-{max(f_p0):.3f}, mean "
        f"{sum(f_p0)/len(f_p0):.3f}), control n={len(c_p0)} (p0 "
        f"{min(c_p0):.3f}-{max(c_p0):.3f}, mean {sum(c_p0)/len(c_p0):.3f})"
        + (f", nearrel n={len(n_p0)} (mean {sum(n_p0)/len(n_p0):.3f})"
           if n_p0 else "")
        + "; registered-control relations disjoint from the facts "
        "(capitals/languages/currencies of countries vs company-founders/"
        "products/mythology); nearrel shares the fact TEMPLATE over "
        "disjoint entities (the locus check).")
    metrics["form_matching"] = form_match
    log(form_match)

    # ------------------------------------------- P2 baseline readouts (t=0)
    def state_record(step, fb, cb, nb, hp, ce=None):
        """The uniform per-state record (journal + metrics shape)."""
        def bat(b):
            return {"mean_p": b["mean_p"], "frac_top1": b["frac_top1"],
                    "frac_top5": b["frac_top5"],
                    "probes": {r["fact"]: {"p": r["p"], "rank": r["rank"]}
                               for r in b["probes"]}}
        return {"step": step, "fact": bat(fb), "ctrl": bat(cb),
                "near": bat(nb) if nb is not None else None,
                "ppl": hp["ppl"], "ce": hp["ce"], "in_batch_ce": ce}

    base_fact = probe_battery(net0, battery)
    base_ctrl = probe_battery(net0, cbattery)
    base_near = probe_battery(net0, nbattery) if nbattery else None
    base_hp = ppl_eval(net0, *bank_xy)
    G_BATT_FACT["R0_replay_kept20"] = base_fact["mean_p"]
    G_BATT_FACT["R0_dp"] = abs(base_fact["mean_p"] - e_base["mean_p"])
    G_BATT_FACT["bank_ppl_replay"] = base_hp["ppl"]
    G_BATT_FACT["bank_ppl_e182"] = e_base["bank_ppl"]
    G_BATT_FACT["pass"] = bool(
        G_BATT_FACT["kept_set_equal"]
        and G_BATT_FACT["max_per_probe_dp"] <= TOL_BASE_PROBE_DP
        and G_BATT_FACT["R0_dp"] <= TOL_BASE_R0_DP
        and abs(base_hp["ppl"] / e_base["bank_ppl"] - 1.0)
        <= TOL_BASE_PPL_REL)
    log(f"G_BATT_FACT @t0: battery R0 {base_fact['mean_p']:.4f} vs e182 "
        f"{e_base['mean_p']:.4f} (dp {G_BATT_FACT['R0_dp']:.5f}) | bank ppl "
        f"{base_hp['ppl']:.2f} vs {e_base['bank_ppl']:.2f} -> "
        f"{'PASS' if G_BATT_FACT['pass'] else 'FAIL'}")
    states = [state_record(0, base_fact, base_ctrl, base_near, base_hp)]
    log(f"t=0: fact R0 {base_fact['mean_p']:.4f} (top1 "
        f"{base_fact['frac_top1']:.2f}) | control R0 "
        f"{base_ctrl['mean_p']:.4f} (top1 {base_ctrl['frac_top1']:.2f})"
        + (f" | nearrel R0 {base_near['mean_p']:.4f}"
           if base_near else "")
        + f" | bank ppl {base_hp['ppl']:.2f} (e182 {e_base['bank_ppl']:.2f})")

    metrics["gates"] = {"G_STATES": G_STATES, "G_SIZE": G_SIZE,
                        "G_BATT_FACT": G_BATT_FACT,
                        "G_REPRO_CORPUS": G_REPRO_CORPUS,
                        "G_CTRL": {k: v for k, v in G_CTRL.items()
                                   if k != "drops"}}
    metrics["corpus"] = corpus_stats
    metrics["states"] = states
    journal_path.write_text(json.dumps({"states": states}, indent=1),
                            encoding="utf-8")
    write_metrics("PARTIAL: batteries built, t=0 read; replay pending")

    # ------------------------------------------- P3/P4 the replay + readouts
    resume = None
    if latest_ck.exists() and not SMOKE:
        try:
            resume = torch.load(latest_ck, map_location=CPU)
            jraw = json.loads(journal_path.read_text(encoding="utf-8"))
            jstates = jraw["states"]
            assert (0 in {s["step"] for s in jstates}
                    and max(s["step"] for s in jstates) == resume["step"]), \
                "journal/checkpoint step mismatch"
            states = jstates
            log(f"journal: {len(states)} states restored "
                f"(steps {[s['step'] for s in states]})")
            metrics["states"] = states
            metrics["resumed_from_step"] = int(resume["step"])
        except Exception as e:  # noqa: BLE001
            log(f"resume failed ({e}); starting the replay from scratch")
            resume = None
            states = [states[0]]
    done_steps = {s["step"] for s in states}

    def on_ckpt(step, sd, ce):
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        fb = probe_battery(evl, battery)
        cb = probe_battery(evl, cbattery)
        nb = probe_battery(evl, nbattery) if nbattery else None
        hp = ppl_eval(evl, *bank_xy)
        del evl
        rec = state_record(step, fb, cb, nb, hp, ce)
        states.append(rec)
        states.sort(key=lambda s: s["step"])
        journal_path.write_text(json.dumps({"states": states}, indent=1),
                                encoding="utf-8")
        metrics["states"] = states
        log(f"  CKPT +{step:3d}: fact {fb['mean_p']:.4f} (ret "
            f"{fb['mean_p'] / base_fact['mean_p']:.3f}) | control "
            f"{cb['mean_p']:.4f} (ret "
            f"{cb['mean_p'] / base_ctrl['mean_p']:.3f})"
            + (f" | near {nb['mean_p']:.4f}" if nb else "")
            + f" | ppl {hp['ppl']:.2f} | in-batch CE {ce:.4f}"
            + (f" | e182 fact {e_traj[step]['mean_p']:.4f} ppl "
               f"{e_traj[step]['bank_ppl']:.2f}" if step in e_traj else ""))
        write_metrics(f"PARTIAL: states probed through +{step}")

    todo = tuple(s for s in CK_MAIN if s > max(done_steps))
    if todo:
        log(f"THE REPLAY: lr {LR}, {todo[-1]} steps total, batch {BATCH} x "
            f"ctx {SEQ}, AdamW (0.9,0.95) wd 0.1 clip 1.0, seed "
            f"{FREEZE_SEED} (e182's own), checkpoints +{list(todo)}")
        fin = replay_wash(net0, train_ids, bank_xy, todo, on_ckpt,
                          resume=resume)
        log(f"replay finished at step {fin}")

    # ------------------------------------------- verification gate G_REPLAY
    vr = {"shared_steps": shared_steps, "per_step": {}, "tol_dp":
          TOL_REPLAY_DP, "tol_ppl": [TOL_REPLAY_PPL_LO,
                                     TOL_REPLAY_PPL_HI]}
    ok = not SMOKE
    st = {s["step"]: s for s in states}
    for s_ in shared_steps:
        d_mean = abs(st[s_]["fact"]["mean_p"] - e_traj[s_]["mean_p"])
        r_ppl = st[s_]["ppl"] / e_traj[s_]["bank_ppl"]
        vr["per_step"][str(s_)] = {
            "fact_mean_p": [st[s_]["fact"]["mean_p"],
                            e_traj[s_]["mean_p"]],
            "dp": d_mean, "ppl": [st[s_]["ppl"], e_traj[s_]["bank_ppl"]],
            "ppl_ratio": r_ppl,
            "in_batch_ce": [st[s_]["in_batch_ce"],
                            e_traj[s_]["in_batch_ce"]]}
        ok = ok and d_mean <= TOL_REPLAY_DP \
            and TOL_REPLAY_PPL_LO <= r_ppl <= TOL_REPLAY_PPL_HI
    f50 = 1 - st[50]["fact"]["mean_p"] / st[0]["fact"]["mean_p"] \
        if 50 in st else None
    vr["fact_decline_at_50"] = f50
    vr["shape_tol_decl50"] = TOL_REPLAY_SHAPE_DECL50
    if not SMOKE:
        ok = ok and f50 is not None and f50 >= TOL_REPLAY_SHAPE_DECL50
    vr["pass"] = bool(ok)
    G_REPLAY = vr
    metrics["gates"]["G_REPLAY"] = G_REPLAY
    metrics["gates"]["G_PPL"] = {
        "rule": "bank ppl read at every state (the improving reference)",
        "ppl_curve": {str(s["step"]): s["ppl"] for s in states},
        "ppl_improves": bool(states[-1]["ppl"] < states[0]["ppl"]),
        "pass": True}

    # ------------------------------------------- P5 adjudication (frozen)
    R0f = st[0]["fact"]["mean_p"]
    R0c = st[0]["ctrl"]["mean_p"]
    deepest = max(st)

    def decl(step):
        return {"fact": 1 - st[step]["fact"]["mean_p"] / R0f,
                "ctrl": 1 - st[step]["ctrl"]["mean_p"] / R0c}

    d_star = decl(deepest)
    d_50 = decl(50) if 50 in st else None
    R0n = st[0]["near"]["mean_p"] if st[0].get("near") else None
    near_decl = {str(s["step"]): 1 - s["near"]["mean_p"] / R0n
                 for s in states if s["step"] > 0 and s.get("near")} \
        if R0n else {}

    def bars(d):
        fact_decl, ctrl_decl = d["fact"], d["ctrl"]
        fact_specific = bool(ctrl_decl <= HOLD_DECL
                             and fact_decl >= ERODE_DECL)
        forgetting_generic = bool(
            (not fact_specific) and fact_decl > GEN_FLOOR_FACT_DECL
            and fact_decl / GEN_BAND <= ctrl_decl <= fact_decl * GEN_BAND)
        verdict = ("FACT-SPECIFIC" if fact_specific else
                   "FORGETTING-GENERIC" if forgetting_generic else "MIXED")
        return fact_specific, forgetting_generic, verdict

    fs, fg, verdict = bars(d_star)
    fs50, fg50, verdict50 = bars(d_50) if d_50 else (None, None, None)
    ratio = (d_star["ctrl"] / d_star["fact"]
             if abs(d_star["fact"]) > 1e-9 else None)
    verification_ok = bool(G_BATT_FACT["pass"] and G_REPRO_CORPUS["pass"]
                           and G_REPLAY["pass"])
    failed_gates = [g for g, v in (("G_BATT_FACT", G_BATT_FACT["pass"]),
                                   ("G_REPRO_CORPUS",
                                    G_REPRO_CORPUS["pass"]),
                                   ("G_REPLAY", G_REPLAY["pass"]))
                    if not v]

    if not verification_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (curves reported; no bar read)"
        clause = ("verification gates failed: "
                  + ", ".join(failed_gates)
                  + " — the states here are NOT certified as e182's; no "
                  "bar adjudicated (a certified replay of steps 1-25 needs "
                  "the original GPU, i.e. a phase-2 re-run).")
    elif verdict == "FACT-SPECIFIC":
        clause = (f"controls hold <= 20% (decline {d_star['ctrl']:.3f}) "
                  f"while the fact erodes >= 50% (decline "
                  f"{d_star['fact']:.3f}) at +{deepest} — fact-specific; "
                  "the surgical signature STRENGTHENS; phase 2 (fresh "
                  "corpus draws, GPU) unlocks")
    elif verdict == "FORGETTING-GENERIC":
        clause = (f"controls erode within ~1.5x of the fact probes' "
                  f"relative decline at +{deepest} (ctrl "
                  f"{d_star['ctrl']:.3f} vs fact {d_star['fact']:.3f}, "
                  f"ratio {ratio:.2f}; band "
                  f"[{d_star['fact'] / GEN_BAND:.3f}, "
                  f"{d_star['fact'] * GEN_BAND:.3f}]) — T123's erosion is "
                  "generic forgetting at 124M; the GPT-2 clause scopes to "
                  "'ordinary forgetting with improving perplexity', NOT a "
                  "no-basin signature")
    else:
        part50 = ("; at +50: fact %.3f vs control %.3f"
                  % (d_50["fact"], d_50["ctrl"])) if d_50 else ""
        clause = (f"curves verbatim; the scope sentence carries numbers "
                  f"(at +{deepest}: fact decline {d_star['fact']:.3f}, "
                  f"control decline {d_star['ctrl']:.3f}, ratio "
                  + (f"{ratio:.2f}" if ratio is not None else "n/a")
                  + part50 + ")")
    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"

    gates_summary = {"G_STATES": G_STATES["pass"], "G_SIZE": G_SIZE["pass"],
                     "G_BATT_FACT": G_BATT_FACT["pass"],
                     "G_REPRO_CORPUS": G_REPRO_CORPUS["pass"],
                     "G_CTRL": G_CTRL["pass"], "G_REPLAY": G_REPLAY["pass"]}
    metrics["adjudication"] = {
        "bars": {"FACT_SPECIFIC": fs, "FORGETTING_GENERIC": fg,
                 "verdict": verdict, "clause": clause,
                 "order": "FACT-SPECIFIC -> FORGETTING-GENERIC -> MIXED "
                          "(gated on G_REPLAY)"},
        "bar_constants": {"GEN_BAND": GEN_BAND,
                          "GEN_FLOOR_FACT_DECL": GEN_FLOOR_FACT_DECL,
                          "HOLD_DECL": HOLD_DECL, "ERODE_DECL": ERODE_DECL},
        "declines_at_deepest": {**d_star, "step": deepest,
                                "decline_ratio_ctrl_over_fact": ratio},
        "co_adjudication_at_50": None if not d_50 else {
            "declines": d_50, "FACT_SPECIFIC": fs50,
            "FORGETTING_GENERIC": fg50, "verdict": verdict50,
            "note": "e182's own deepest MEASURED checkpoint (the T123-stamp "
                    "comparison); the PRIMARY verdict is the deepest probed "
                    "state"},
        "declines_per_state": {str(s["step"]): decl(s["step"])
                               for s in states if s["step"] > 0},
        "nearrel_co_report": {
            "declines": near_decl,
            "note": "same-template (US-state capitals) co-report: if nearrel "
                    "erodes with the facts while the registered controls "
                    "hold, the decay is template/capital-relation-level, "
                    "not knowledge-level; NOT part of the frozen bars",},
        "e182_fact_mean_p": {str(k): v["mean_p"] for k, v in e_traj.items()},
        "gates_summary": gates_summary,
        "fact_facts": [r["fact"] for r in battery],
        "ctrl_facts": [r["fact"] for r in cbattery],
        "nearrel_facts": [r["fact"] for r in nbattery],
    }
    log("=" * 78)
    log(f"E182C VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")

    metrics["honesty_reflex"] = {
        "phase_1": "matched batteries under ONE frozen wash trajectory; "
                   "phase 2 (>=2 fresh corpus draws, GPU) owns generality",
        "states_regenerated": "e182 saved no wash states; the states here "
                              "are a CPU fp32 replay of the frozen recipe, "
                              "admitted by G_REPLAY (steps 1-25 were GPU in "
                              "e182 — the measured deviation is the "
                              "deferred CPU/GPU-numerics item, now "
                              "quantified)",
        "control_guarantees_nothing": "controls holding does not prove the "
                                      "facts special (weaker consolidation "
                                      "is an alternative); controls eroding "
                                      "does not prove generic (shared "
                                      "few-shot locus, shared prompt "
                                      "family); the openness is the point",
        "single_seed_single_corpus": "one organism, one wash seed (e182's "
                                     "18202), one frozen corpus draw",
        "few_shot_conflation_inherited": "the 2-shot context is part of the "
                                         "frozen instrument; decay's locus "
                                         "(fact storage vs in-context "
                                         "task-following) remains "
                                         "unadjudicated — the nearrel "
                                         "co-report is the partial check "
                                         "(shared template, disjoint "
                                         "entities), not a discharge",
        "exemplar_pair_sensitivity": "control p0 depends on the fixed "
                                     "exemplar pair (observed swings 0.12-"
                                     "0.55 across screening orders); the "
                                     "pool order was frozen after "
                                     "pristine-model screening and never "
                                     "tuned again",
        "control_domain_caveat": "the registered controls are brand/"
                                 "corporate knowledge (the family that "
                                 "passes the gate at 124M); 'controls hold' "
                                 "may not generalize to all non-fact "
                                 "knowledge — phase 2's fresh draws owe the "
                                 "breadth",
        "plus80_never_probed_by_e182": "e182's measurements stop at +50; "
                                       "+80 is its true final step, probed "
                                       "here first",
        "baseline_matching": "both batteries share the gate and cap; "
                             "baseline p distributions co-reported in "
                             "form_matching (decline is per-battery "
                             "relative)",
        "selection_bias": "the control battery is the best-known members of "
                          "a hand-curated pool (the same top-of-passers "
                          "bias e182's battery has)",
    }
    metrics["compute"] = {
        "envelope": "CPU-only (124M fp32 inference + the 80-step frozen "
                    "replay, batch 8x512, threads "
                    f"{torch.get_num_threads()}); no GPU claim",
        "walls_s_per_step_logged": True,
        "state_archive": [f"runs/checkpoints/{NAME}_s{s}.pt"
                          for s in CK_MAIN if s in st],
        "resumable": f"runs/checkpoints/{NAME}_replay_latest.pt",
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    if deepest < CK_MAIN[-1] and not SMOKE:
        trims.append(f"replay trimmed at +{deepest} (intended "
                     f"+{CK_MAIN[-1]}) — deepest-reached state adjudicated, "
                     "DISCLOSED")
    write_metrics(("DONE" if deepest >= CK_MAIN[-1]
                   else f"DONE (trimmed at +{deepest})")
                  if not SMOKE else "SMOKE DONE")
    if not SMOKE:
        png = make_plot(rd, states, verdict, clause,
                        metrics["adjudication"], form_match)
        log(f"outputs: {rd / 'metrics.json'}, {png}")
    else:
        png = None
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
