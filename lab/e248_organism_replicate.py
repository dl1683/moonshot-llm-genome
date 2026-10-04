"""E248 — THE ORGANISM REPLICATE (the R64 auditor's dispatch; the first
multi-phase owner-lane cell of the era; e248-replicate identity).

WHY (scratch/e244_organism_replicate_sketch.md, option (a); scratch/
review_r64_auditor.md item (a)): the day's entire 124M program rides ONE
organism (the GPT-2-derived wash archive). The triangle's instruments
are validated (e234/e238/e239/e237; T223's closed book) and the
max-priority compute window is open, so the true replicate is now
affordable: a FRESH ~124M-class char-LM trained under the window, the
fact-install, the battery suite, TWO washes, and the eight committed
bars re-read at matched conventions. NO new instruments — the bars are
known; the sketch deliberately names none of its own. The small worlds
(e242 et al.) are the cross-scale story, NOT substitutes (the sketch's
own words).

BUILD ONS (Rule 6b; every instrument a named re-read):
  bar 1 P-FIRST census   <- e230's conventions (runs/e230/metrics.json;
                            margin_sigma < 0.05 flip zone; p < 0.5*p_t0;
                            absorbing first-crossing classes)
  bar 2 the seat         <- e226's conventions (support = grad p(answer|
                            prompt) over all params, L2-normalized; wash
                            gradient at the state's own next draw;
                            Z vs the product-family band; the dying-
                            direction split 2.6-14.8x)
  bar 3 T(t)             <- e238's conventions (one-T Bernoulli MLE over
                            the battery p-side under q_i(T)=softmax(L0_i/
                            T)[answer]; R2 vs the t0-decline denominator)
  bar 4 the span         <- e234's conventions (split-half 2-dim Gram-SVD
                            span of the wash's step gradients; cross-wash
                            top-basis cos; committed 0.96)
  bar 5 zombie rates     <- e232's conventions (standing at +80 = margin
                            >= 0.05 with p < half; per-cell standing
                            fractions; the modal ZOMBIES-STAND rule)
  co 6 common-mode       <- e239's conventions (theta-ratio envelope; 1 -
                            median theta_between/(theta1+theta2))
  co 7 anchor exposure   <- e234's wind_cum (cumulative |cos(P_span g_t,
                            s_i)| over the decomposition half)
  co 8 residual-order    <- e238's blocked Spearman (thermal residual vs
                            the same-organism committed erosion order)
  the wash recipe        <- e182c/e182c2 VERBATIM (AdamW (0.9,0.95) wd
                            0.1 constant lr 5e-5, clip 1.0, batch 8 x
                            ctx 512, seeded window-draw stream, +80
                            steps, states {2,10,50,80}/{10,50,80})
  the battery            <- e182/e182c/e182c2 pools VERBATIM (the
                            committed 54: fact 20 / ctrl 12 / near 3 /
                            tmpl 19), 2-shot fixed-exemplar construction,
                            the e182 probe gate at installed t0
  the organism chassis   <- lab/common.py TinyGPT, the g-series recipe
                            scaled (AdamW (0.9,0.95) wd 0.1, lr 1e-3,
                            cosine + warmup, clip 1.0), g1bS's
                            val-anchored-cosine + U-turn lesson

WHAT IS NEW: the organism (a fresh 116,279,296-param char-LM — "~124M-
class", disclosed exactly; 12 layers / 14 heads / 896 embd / 512 ctx /
65-char vocab, the GPT-2-small depth at char level), the install phase
(the e182c-family fine-tuning recipe PORTED to the teaching direction
with a frozen dose ladder), the char-level battery dialect (the answer
read at its FIRST CHARACTER after a frozen trailing-space prompt), and
the two fresh washes on their own draw seeds. Everything else is a
verbatim re-read.

===== THE REGISTRATION (frozen BEFORE any compute; no bar shopping) =====

THE FIVE PRIMARY BARS (each a re-read of a committed number, with its
tolerance frozen here):
  (1) P-FIRST-MODALITY — the e230 census on the new organism's two
      washes. Classes verbatim: margin event = first post-t0 state with
      argmax margin_sigma < 0.05 (absorbing); p event = first post-t0
      state with p < 0.5*p_t0 (absorbing); P-FIRST = p crossed strictly
      before the margin; MARGIN-FIRST = margin strictly before p;
      TOGETHER = same state; NEITHER. Committed: P-FIRST is the largest
      crossing class in every battery-wash (fact-w1 8/20, fact-w2 8/20,
      ctrl-w1 4/12, ctrl-w2 3/12, tmpl-w1 9/19, tmpl-w2 9/19; near 3/3
      co-report). BAR: P-FIRST count > max(MARGIN-FIRST, TOGETHER)
      count in >= 4 of the 6 adjudicated cells (fact/ctrl/tmpl x
      w1'/w2'; the e232 modal rule at its own 4-of-6 line; near never
      adjudicates).
  (2) THE SEAT'S Z-DIRECTION — the e226 anchor read on the new
      organism's own fate-split pair. Anchors frozen-rule: per wash, the
      product family's (make-relation, n=7) max-hr80 probe = the holder,
      min-hr80 = the dying (hr = p(+80)/p(0)); the wash's split is real
      only if holder_hr/dying_hr >= 2.0 (committed GPT-2 w1: 0.905/
      0.184 = 4.9). Registered reads: A(w,s,i) = cos(g(w,s), s_i(0)) at
      states {0,2,10,50,80} x 2 washes; sigma = std(ddof=1) of the
      product family's 5 non-anchor probes at that read; Z = |A_holder
      - A_dying|/sigma. BAR (both clauses): (a) Z >= 2.0 at >= 1
      registered read (e226's SUPPORT-DIFFERENTIATES line verbatim);
      (b) the dying-direction split |A_dying|/|A_holder| >= 2.0 at >= 2
      of the 6 mid/settled reads (states {10,50,80} x 2 washes;
      committed 2.6-14.8x across three washes). The t=0 inversion
      (|A_holder| > |A_dying| at s=0) is a CO-READ, never adjudicated.
      If neither wash carries a >= 2x fate split, bar (2) reads
      NO-SPLIT (a finding, not a pass).
  (3) T-R2 IN [0.5, 0.8] — the e238 thermal fit re-read: per wash x
      deep state {+50, +80}, the one-T Bernoulli MLE over the pooled 54
      probes' observed p under q_i(T) = softmax(L0_i/T)[answer char],
      L0_i = the installed-t0 full answer-position logits; R2 = 1 -
      sum(p_obs - q)^2 / sum(p_obs - p0)^2. Committed: {w1 0.626/0.690,
      w2 0.600/0.709}. BAR: R2_pooled in [0.5, 0.8] at >= 2 of the 4
      deep cells with BOTH washes represented; T(t) trajectory co-read
      (committed 1.00 -> 1.45 monotone-ish).
  (4) THE SPAN'S CROSS-WASH IDENTITY > 0.9 — the e234 span re-read: per
      wash, the top-2 span basis by Gram-SVD (fp64 Gram of the
      L2-normalized step gradients, steps 1..40; split-half: steps
      41..80 decomposed against it); identity = |cos| of the TOP basis
      directions across the two washes (sign-aligned). Committed: top
      dim 0.962/0.966/0.958 (2nd dim ~0.885 co-read). BAR: top-dim
      identity > 0.9.
  (5) THE ZOMBIE RATES WITHIN 2X — the e232 lag census re-read:
      population = the cell's P-FIRST probes; STANDING at +80 =
      margin_sigma(+80) >= 0.05 AND p(+80) < 0.5*p_t0; RESOLVED =
      margin(+80) < 0.05. Committed standing fractions {fact 0.75/0.75,
      ctrl 1.00/1.00, tmpl 0.67/0.89}. BAR: >= 4 of 6 adjudicated cells
      have standing_fraction in [0.5x, 2x] of the committed cell value
      (capped at 1.0) AND the modal verdict replicates (>= 4 of 6 cells
      standing_fraction >= 0.5 = ZOMBIES-STAND).

THE THREE CO-BARS (the auditor's amendments; each barred):
  (6) THE TIDE'S COMMON-MODE COHERENCE >= 0.5, BOTH LEGS — the e239
      envelope re-read: coherence(t) = 1 - median_i[theta_between_i /
      (theta1_i + theta2_i)] where theta_between_i = angle(s_i(w1',t),
      s_i(w2',t)), theta1/theta2 = angles to s_i(0). Committed: 0.873
      (+50) / 0.839 (+80) (the "84-87%" range, R64-corrected).
      BAR: coherence >= 0.5 at BOTH states {+50, +80} (both legs).
  (7) THE ANCHOR EXPOSURE RATIO ~5-10x — the e234 wind_cum re-read:
      wind_cum(w,i) = sum over the decomposition half's steps (41..80)
      of |cos(P_span g_t, s_i(0))|; ratio = wind_cum(dying)/
      wind_cum(holder) per wash (the frozen-rule anchors of bar 2).
      Committed: 7.7/9.0/6.5 (R64-corrected from the escaped 8-10x).
      BAR: ratio in [5, 10] in >= 1 wash (both washes co-reported).
  (8) THE RESIDUAL-ORDER JOIN — the e238 residual structure re-read:
      per wash x deep state, blocked Spearman (within-battery ranks
      pooled over fact/ctrl/tmpl) between the per-probe thermal
      residual r_i = p_obs,i - q_i(T*) and the same organism-wash's
      erosion order (erosion_i = 1 - p(+80)/p(0), the e214
      convention). Committed: -0.898 pooled (w1+50; e238's own bar
      0.4; the dispatch's 0.5).
      [PHASE-BOUNDARY AMENDMENT, pre-wash-compute, the coordinator's
      R64-critic-sourced advisory — STRENGTHENING, disclosed in
      metrics.deviations]: the phase-1c smoke showed this rank
      statistic passes a NULL (pure-temperature synthetic truth fires
      |rho| up to 0.812, sign-flipping; the deterministic seed-0
      re-measurement supersedes the executor's earlier ~0.67 quote —
      the advisory's 0.75 line was set from that quote, so the line
      follows the MEASURED floor: 0.85). BAR (both clauses, per cell):
      (a) |rho_blocked| >= 0.85 AND (b) mean_abs_resid > 2.443e-8
      (10x the synthetic null's max magnitude 2.443e-9; the committed
      real magnitudes 0.049-0.216 sit ~9 orders above). PASS := >= 2
      of the 4 deep cells clear BOTH clauses; sign co-reported.

THE GATES (any failure = INSTRUMENT-DEAD, the full autopsy):
  G_ORGANISM  — the base char-LM's val CE <= 1.50 nats/char (the lab's
                healthy range for shakespeare char-LMs; target <= 1.40);
                param count recorded exactly; the val curve monotone-ish
                with the U-turn guard honored.
  G_SIZE      — params <= 500M with the stated reason: the replicate
                demands the 124M organism class (the dispatch's own
                sizing, option (a)); 116,279,296 disclosed.
  G_RECALIBRATION [AMENDMENT item (i), phase-boundary] — the census
                thresholds (the 0.05-sigma flip zone; the 0.5*p_t0
                halving line) are a PORT from the BPE organism: verify
                at installed t0 that the fresh char-level organism's
                margin mass lives on a scale where they bite (median
                margin_sigma >= 0.10, p25 >= 0.05, median p0 in
                [0.30, 0.999]). PASS -> verbatim thresholds (a true
                port). FAIL -> RECALIBRATE the flip zone to
                0.05 * (median_t0_margin / 0.82) (the committed
                archive's P-FIRST-pooled t0 margin median, the scale
                anchor), the halving line unchanged (p-relative),
                DISCLOSED; the census adjudicates on the recalibrated
                zone with the verbatim-zone census co-reported.
  G_INSTALL   — >= 6 probes per battery pass the e182 gate at the
                installed t0 ((top-1 and p >= 0.8) or (top-5 and
                p >= 0.5), the char-dialect answer read); the install
                dose ladder fully disclosed; wash-corpus CE within
                +10% of the base organism's on the same held-out bank
                (the install must not wreck the LM).
  G_CORPUS    — the wash corpus = data/input.txt line-filtered by the
                banned list (every battery subject + answer string,
                case-insensitive); the filter stats recorded; the
                contamination scan ZERO for every kept probe.
  G_DRAWS     — each wash's draw stream archived (seed + generator
                state at +80 saved with every state file).
  G_REPRO     — the saved t0 re-probes its own journal rows
                (max |dp| <= 0.005 on re-read; single deterministic
                evals).
  G_SUPPORTFD — e226's directional gate ported: p(theta + 0.02*s_hat)
                > p0 for every probe support used; CE(W - 0.002*g_hat)
                < CE(W) for every wash-gradient read used.

THE LADDERS (pre-registered contingencies, frozen here):
  WASH-DEPTH LADDER — the primary wash grid is {2,10,50,80} (w1') /
      {10,50,80} (w2') at the VERBATIM e182 recipe (lr 5e-5 constant,
      80 steps). IF fact_decl(deepest) < 0.10 (the wash too gentle for
      the fresh organism's entrenchment), re-run BOTH washes at depth
      x2 (160 steps, grids {4,20,100,160}/{20,100,160}) then x4 (320
      steps, {8,40,200,320}/{40,200,320}), same recipe, same seeds,
      every rung disclosed. ADJUDICATION uses the FIRST rung where
      fact_decl(deepest) >= 0.25 (a real death to read); if no rung
      reaches it by x4, the cell adjudicates at x4 with the shallowness
      disclosed (the census reads the deepest rung; bars (1)/(5) carry
      the thin-death caveat).
  INSTALL DOSE LADDER — the install = continued training from the base
      organism on the TEACH stream (the wash corpus with the 54 battery
      statements spliced every 150 chars) under the VERBATIM e182
      recipe (lr 5e-5, 80-step passes). Dose rungs: passes {1,2,4,8} at
      lr 5e-5, then lr {1e-4, 2e-4} x passes {2,4}. STOP at the first
      rung where G_INSTALL passes AND the battery-mean installed p0 is
      in [0.55, 0.97]. If no rung passes by the last, INSTRUMENT-DEAD
      at G_INSTALL.

THE VERDICTS (phase 5): REPLICATES (all five primaries hold within
their frozen tolerances — the laws draft's constants are organism-
robust; W038's bands are set) / PARTIAL (name which bars failed — each
failure is a finding about which objects are organism-specific) /
INSTRUMENT-DEAD (any gate fails — the full autopsy). No bar shopping;
the tolerances frozen above.

HONESTY (carried on every read): n=1 fresh organism; the committed
archive was GPT-2/BPE/web-corpus — this cell is char-level/shakespeare-
derived, so the bars are INSTRUMENT re-reads at matched conventions,
not distribution-matched replications; landing outside tolerance reads
organism-specificity, and the verdict vocabulary is about the reads.
[AMENDMENT item (ii)] THE SPAN READ'S SIGN-SHADOW GUARD: e234's
Gram-SVD span was validated on GPT-2's washes only; the two fresh
washes share corpus/optimizer/lr (draw-independence only), so a
cross-wash identity pass reads "recipe+organism" at minimum — the
optimizer's sign-pattern geometry (g12) is a candidate shadow the
identity bar cannot alone exclude; the corpus-swap replication is the
stronger test, named, not run (T216's honesty clause, inherited).
The t=0 anchoring disclosure of the whole 124M program (one organism)
is discharged by this cell existing at all. Nothing guaranteed.

COMPUTE (owner directive ACTIVE at dispatch: max-priority window,
bursts <= 180 s, cooldowns 30-60 s, PER-STEP THERMAL POLLS at a 78C
margin — max 82C, zero >= 84C, the e237 discipline; NO concurrent GPU
jobs: e246 owned the GPU lane at registration; phase 2 waits for its
process to clear). GPU burst guard: start a burst only at temp <= 70C;
poll during bursts; break at >= 78C; hard-stop at >= 82C; resume from
checkpoint. Journal resumable (the e233/e237 discipline): every phase
boundary commits + pushes with the (e248-replicate) identity.

Usage (one phase per invocation, from repo root):
  python lab/e248_organism_replicate.py freeze      # phase 1: registration
  python lab/e248_organism_replicate.py build       # desk: corpus+teach+battery
  python lab/e248_organism_replicate.py train       # phase 2 (GPU, after e246)
  python lab/e248_organism_replicate.py install     # phase 3 (GPU)
  python lab/e248_organism_replicate.py wash        # phase 4 (GPU)
  python lab/e248_organism_replicate.py read        # phase 4b: the reads
  python lab/e248_organism_replicate.py adjudicate  # phase 5 (desk)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs" / "e248"
CKPT = REPO / "runs" / "checkpoints"
SCRATCH = REPO / "scratch" / "e248_cache"
sys.path.insert(0, str(REPO / "lab"))

RUN.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

E248_ID = "e248-replicate"
CHUNK = 4_000_000          # column chunk for all big dots (elems)

# ------------------------------------------------------------------ constants
SEED_BASE = 24801                    # the organism's training stream seed
SEED_INSTALL = 24807                 # the install teach-stream seed
SEED_WASH = {"w1": 24811, "w2": 24812}
VAL_SEED = 24803

CFG = dict(vocab=65, n_layer=12, n_head=14, n_embd=896, block_size=512)
PARAMS_EXPECTED = 116_279_296
SIZE_REASON = ("the replicate demands the 124M organism class (dispatch "
               "option (a), scratch/e244 sketch); 116,279,296 params > the "
               "100M free tier, <= the 500M ceiling with this stated reason")

TRAIN = dict(steps=4000, lr=1e-3, warmup=300, batch=32,
             eval_every=200, burst_s=175.0, poll_every=4,
             temp_start_c=66.0, temp_break_c=76.0, temp_hard_c=82.0,
             cooldown_target_c=62.0, cooldown_max_s=300.0,
             amp=True,
             uturn_tol=0.03, val_ce_gate=1.50, val_ce_target=1.40,
             max_bursts=60)

WASH = dict(lr=5e-5, betas=(0.9, 0.95), wd=0.1, clip=1.0, batch=8,
            steps=80, grids={"w1": [2, 10, 50, 80], "w2": [10, 50, 80]},
            burst_s=175.0, poll_every=1, temp_start_c=70.0,
            temp_break_c=78.0, temp_hard_c=82.0)
WASH_LADDER = [1, 2, 4]
DEPTH_SOFT = 0.10
DEPTH_NEED = 0.25

INSTALL = dict(passes_ladder=[1, 2, 4, 8], lr_ladder=[5e-5, 1e-4, 2e-4],
               passes_at_high_lr=[2, 4], splice_gap=150,
               gate_top1=0.8, gate_top5=0.5, floor=6,
               p0_lo=0.55, p0_hi=0.97, ce_wreck_tol=0.10)

BARS = dict(
    pfirst_cells_need=4,
    seat_z_line=2.0, seat_ratio_line=2.0, seat_ratio_reads_need=2,
    seat_fate_split=2.0,
    tr2_lo=0.5, tr2_hi=0.8, tr2_cells_need=2,
    span_identity_line=0.9,
    zomb_factor=2.0, zomb_cells_need=4, zomb_stand_line=0.5,
    coherence_line=0.5,
    exposure_lo=5.0, exposure_hi=10.0, exposure_washes_need=1,
    resid_rho_line=0.85, resid_cells_need=2,
    resid_mag_floor=2.4430e-08,
    recal_margin_med_min=0.10, recal_margin_p25_min=0.05,
    recal_p_lo=0.30, recal_p_hi=0.999,
    committed_anchor_margin_median=0.82,
)
COMMITTED = dict(
    pfirst_counts={"fact-w1": 8, "fact-w2": 8, "ctrl-w1": 4, "ctrl-w2": 3,
                   "tmpl-w1": 9, "tmpl-w2": 9},
    pfirst_ns={"fact": 20, "ctrl": 12, "tmpl": 19},
    seat_ratios="2.6-14.8x (three washes)", seat_max_z=7.19,
    tr2={"w1-50": 0.6259, "w1-80": 0.6896, "w2-50": 0.5995, "w2-80": 0.7085},
    span_identity=[0.9619, 0.9661, 0.9581],
    zombie_standing={"fact-w1": 0.75, "fact-w2": 0.75, "ctrl-w1": 1.00,
                     "ctrl-w2": 1.00, "tmpl-w1": 0.667, "tmpl-w2": 0.889},
    coherence={"50": 0.873, "80": 0.839},
    exposure_ratios=[7.7, 9.0, 6.5],
    resid_rho_pooled=-0.898,
    t_of_t="1.00 -> 1.45 monotonish (e238)",
)

# ------------------------------------------------- battery (pools VERBATIM)
CAPS = [("France", "Paris"), ("Greece", "Athens"), ("Poland", "Warsaw"),
        ("Portugal", "Lisbon"), ("Egypt", "Cairo"), ("Ireland", "Dublin")]
LANGS = [("France", "French"), ("Germany", "German"), ("Japan", "Japanese"),
         ("Italy", "Italian"), ("Spain", "Spanish"), ("Russia", "Russian"),
         ("China", "Chinese"), ("England", "English"), ("Greece", "Greek"),
         ("Poland", "Polish")]
CURS = [("Japan", "yen"), ("the United Kingdom", "pound"),
        ("the United States", "dollar"), ("China", "yuan")]
FACT_POOLS = {"cap": CAPS, "lang": LANGS, "cur": CURS}
FACT_TMPL = {
    "cap": ("The capital of {s} is {a}. ", "The capital of {s} is"),
    "lang": ("People in {s} speak {a}. ", "People in {s} speak"),
    "cur": ("The currency of {s} is the {a}. ", "The currency of {s} is the"),
}
CTRL_POOLS = {
    "found": [("The social network founded by Mark Zuckerberg", "Facebook"),
              ("The electric car company founded by Elon Musk", "Tesla"),
              ("The rocket company founded by Elon Musk", "SpaceX"),
              ("The search engine founded by Larry Page", "Google"),
              ("The online encyclopedia that anyone can edit", "Wikipedia")],
    "make": [("The gaming console made by Microsoft", "Xbox"),
             ("The web browser made by Google", "Chrome"),
             ("The phone made by Apple", "iPhone"),
             ("The tablet made by Apple", "iPad"),
             ("The email service made by Google", "Gmail"),
             ("The music store made by Apple", "iTunes"),
             ("The game console made by Sony", "PlayStation")],
}
CTRL_TMPL = ("{s} is called {a}. ", "{s} is called")
NEAR_POOL = [("Massachusetts", "Boston"), ("Georgia", "Atlanta"),
             ("Ohio", "Columbus")]
NEAR_TMPL = ("The capital of {s} is {a}. ", "The capital of {s} is")
TMPL_POOL = [("Boston", "Massachusetts"), ("Atlanta", "Georgia"),
             ("Sacramento", "California"), ("Austin", "Texas"),
             ("Columbus", "Ohio"), ("Denver", "Colorado"),
             ("Little Rock", "Arkansas"), ("Hartford", "Connecticut"),
             ("Dover", "Delaware"), ("Tallahassee", "Florida"),
             ("Honolulu", "Hawaii"), ("Boise", "Idaho"),
             ("Indianapolis", "Indiana"), ("Des Moines", "Iowa"),
             ("Topeka", "Kansas"), ("Baton Rouge", "Louisiana"),
             ("Annapolis", "Maryland"), ("Lansing", "Michigan"),
             ("Jackson", "Mississippi")]
TMPL_TMPL = ("The state whose capital is {s} is {a}. ",
             "The state whose capital is {s} is")


def build_battery() -> list[dict]:
    probes = []

    def add(battery, rel, pool, tmpl, family):
        for i, (s, a) in enumerate(pool):
            ex_idx = ([j for j in range(min(3, len(pool))) if j != i]
                      if i < 2 else [0, 1])
            sent, stem = tmpl
            exs = [sent.format(s=pool[j][0], a=pool[j][1]) for j in ex_idx]
            probes.append(dict(battery=battery, rel=rel, family=family,
                               fact=f"{s}->{a}", subject=s, answer=a,
                               exemplars=exs,
                               sentence=sent.format(s=s, a=a),
                               stem=stem.format(s=s), ans_char=a[0]))

    add("fact", "cap", CAPS, FACT_TMPL["cap"], "cap-cur")
    add("fact", "lang", LANGS, FACT_TMPL["lang"], "lang")
    add("fact", "cur", CURS, FACT_TMPL["cur"], "cap-cur")
    add("ctrl", "found", CTRL_POOLS["found"], CTRL_TMPL, "founder-anchor")
    add("ctrl", "make", CTRL_POOLS["make"], CTRL_TMPL, "product")
    add("near", "cap", NEAR_POOL, NEAR_TMPL, "near-uscap")
    add("tmpl", "rev", TMPL_POOL, TMPL_TMPL, "rev-capital")
    assert len(probes) == 54, len(probes)
    return probes


BATTERY = build_battery()
BANNED = sorted({p["subject"].lower() for p in BATTERY}
                | {p["answer"].lower() for p in BATTERY})


def prompt_of(p: dict) -> str:
    return p["exemplars"][0] + p["exemplars"][1] + p["stem"] + " "


# ------------------------------------------------------------- corpus / chars
_TEXT = (REPO / "data" / "input.txt").read_text(encoding="utf-8")
CHARS = sorted(set(_TEXT))
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
assert len(CHARS) == 65


def encode(s: str) -> list:
    return [STOI[c] for c in s]


def prompt_ids(p: dict) -> list:
    return encode(prompt_of(p))


def ans_index(p: dict) -> int:
    return STOI[p["ans_char"]]


# ------------------------------------------------------------------ io
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jlog(event: str, **kw) -> None:
    rec = {"t": now_iso(), "event": event, **kw}
    with open(RUN / "journal.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(f"[{E248_ID}] {event} "
          + (json.dumps(kw, default=float)[:200] if kw else ""), flush=True)


def load_metrics() -> dict:
    p = RUN / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"experiment": "e248_organism_replicate", "owner_lane": E248_ID,
            "created": now_iso(), "status": "PHASE1-REGISTRATION"}


def write_metrics(m: dict) -> None:
    m["updated"] = now_iso()
    (RUN / "metrics.json").write_text(json.dumps(m, indent=2, default=float),
                                      encoding="utf-8")


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ gpu guard
def gpu_poll() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
             "memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=10).stdout.strip()
        u, mu, mt, t, p = [float(x) for x in out.split(",")]
        return {"util": u, "mem_used": mu, "mem_total": mt, "temp": t,
                "power": p}
    except Exception:
        return {"util": 0, "mem_used": 0, "mem_total": 0, "temp": 0,
                "power": 0}


_PYNVML = {"handle": None, "failed": False}


def temp_fast() -> float:
    """1-5 ms temperature read via pynvml (the 63 ms nvidia-smi subprocess
    cannot outrun this chip's heating: 79->89C in ~2 s of compute at 165W,
    the 19:07 diagnostic's lesson); subprocess fallback."""
    try:
        if _PYNVML["failed"]:
            return gpu_poll()["temp"]
        if _PYNVML["handle"] is None:
            import pynvml
            pynvml.nvmlInit()
            _PYNVML["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
        import pynvml
        return float(pynvml.nvmlDeviceGetTemperature(
            _PYNVML["handle"], pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        _PYNVML["failed"] = True
        return gpu_poll()["temp"]


_PS_SCRIPT = Path(os.environ.get("TEMP", "/tmp")) / "e248_ps.ps1"
_PS_SCRIPT.write_text(
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }\n",
    encoding="utf-8")


def _e246_cell_active() -> str | None:
    """e246 is a MULTI-INVOCATION cell (its agent restarts the python per
    phase) — a process gap is NOT a free lane. The cell is active while
    its metrics.json/run.log mtime is fresh (< 600 s) and its status is
    not DONE. The 2026-10-04 18:23 collision (two 4-step bursts slid
    into an inter-phase gap; 84C hard-stop) is the named failure this
    check exists to prevent."""
    import json as _json
    for p in (REPO / "runs" / "e246" / "metrics.json",
              REPO / "runs" / "e246" / "run.log"):
        try:
            age = time.time() - p.stat().st_mtime
        except OSError:
            continue
        if age < 600.0:
            if p.name == "metrics.json":
                try:
                    st = _json.loads(p.read_text(encoding="utf-8")).get("status", "").upper()
                    if st.startswith("DONE") or st.startswith("COMPLETE"):
                        continue
                except Exception:
                    pass
            return f"e246-cell-active({p.name}, mtime {age:.0f}s ago)"
    return None


def gpu_owner_alive() -> str | None:
    """Live artifacts own the cell: refuse the GPU while e246's cell is
    active (process OR fresh artifacts) or while ANY python holds GPU
    memory. Pure-CPU cells (e.g. e240's archive pass) do not block."""
    cell = _e246_cell_active()
    if cell is not None:
        return cell
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(_PS_SCRIPT)], capture_output=True, text=True,
            timeout=25).stdout
    except Exception:
        out = ""
    gpu_pids = set()
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout
        gpu_pids = {x.strip() for x in q.split() if x.strip().isdigit()}
    except Exception:
        pass
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid = line.split(" ")[0]
        if "e246_engineered_seat" in line:
            return f"e246_engineered_seat(pid {pid})"
        if pid in gpu_pids and "e248_organism_replicate" not in line:
            return f"gpu-holder(pid {pid}: {line[:80]})"
    return None


def thermal_gate(tag: str, start_c: float = 70.0) -> bool:
    owner = gpu_owner_alive()
    if owner is not None:
        jlog("gpu_wait_owner", tag=tag, owner=owner)
        return False
    s = gpu_poll()
    if s["temp"] > start_c:
        jlog("thermal_wait", tag=tag, temp=s["temp"])
        return False
    return True


def mid_burst_check(step, tag, t0, max_s, break_c=78.0,
                    hard_c=82.0) -> tuple[bool, str]:
    tC = temp_fast()
    if tC >= hard_c:
        jlog("thermal_HARDSTOP", tag=tag, step=step, temp=tC)
        return False, "hard"
    if tC >= break_c:
        jlog("thermal_break", tag=tag, step=step, temp=tC)
        return False, "break"
    if time.time() - t0 >= max_s:
        return False, "time"
    return True, ""


PACE = {"sleep": 0.22}


def pace_for_temp(ramp_in: bool = False) -> None:
    """Self-tuning duty controller with hysteresis: holds the chip at
    70-76C (far below the 82C cliff) at the maximum sustainable duty.
    The sleep length grows 1.6x per hot poll (>= 76C), decays 0.8x per
    cool poll (<= 70C), bounded [0.10, 3.0] s. ramp_in: the first steps
    of a burst sleep a little longer so the fans spin up under load."""
    tC = temp_fast()
    if tC >= 76.0:
        PACE["sleep"] = min(PACE["sleep"] * 1.6, 3.0)
    elif tC <= 70.0:
        PACE["sleep"] = max(PACE["sleep"] * 0.8, 0.10)
    s = PACE["sleep"] + (0.30 if ramp_in else 0.0)
    time.sleep(s)


def cooldown(lo=30.0, hi=60.0):
    time.sleep(0.5 * (lo + hi))


def thermal_cooldown(target_c: float = 62.0, max_s: float = 300.0) -> None:
    """Temp-aware cooldown: sleep until <= target_c (or max_s)."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        if gpu_poll()["temp"] <= target_c:
            return
        time.sleep(15.0)
    jlog("cooldown_timeout", temp=gpu_poll()["temp"])


# ------------------------------------------------------------------ phase 1
def cmd_freeze() -> None:
    m = load_metrics()
    m.update({
        "phase": "PHASE1 registration (desk)",
        "status": "REGISTERED",
        "registration": {
            "docstring": "this file's module docstring (the registration, verbatim)",
            "bars": BARS, "committed": COMMITTED,
            "ladders": {"wash_depth": WASH_LADDER,
                        "install_dose": {"passes": INSTALL["passes_ladder"],
                                         "lrs": INSTALL["lr_ladder"]}},
            "verdicts": ["REPLICATES", "PARTIAL", "INSTRUMENT-DEAD"],
        },
        "organism_plan": {"cfg": CFG, "params_expected": PARAMS_EXPECTED,
                          "size_reason": SIZE_REASON},
        "provenance": {"git_head_at_freeze": git_head(),
                       "script": str(Path(__file__).resolve()),
                       "script_sha256_16": sha16(Path(__file__))},
    })
    write_metrics(m)
    jlog("phase1_registration_frozen", primaries=5, co_bars=3)


# ------------------------------------------------------------------ build
def cmd_build() -> None:
    import torch
    lines = _TEXT.split("\n")
    kept, dropped = [], []
    for ln in lines:
        low = ln.lower()
        (dropped if any(b in low for b in BANNED) else kept).append(ln)
    wash_text = "\n".join(kept)
    wash_ids = torch.tensor(encode(wash_text), dtype=torch.long)
    n_train = int(0.9 * len(wash_ids))

    scan = [{"fact": p["fact"],
             "subject_absent": p["subject"].lower() not in wash_text.lower(),
             "answer_absent": p["answer"].lower() not in wash_text.lower()}
            for p in BATTERY]

    stmts = [p["sentence"] for p in BATTERY]
    chunks, i, pos = [], 0, 0
    train_text = wash_text[:n_train]
    while pos < len(train_text):
        end = min(pos + INSTALL["splice_gap"], len(train_text))
        chunks.append(train_text[pos:end])
        chunks.append(stmts[i % len(stmts)])
        i += 1
        pos = end
    teach_text = "".join(chunks)
    teach_ids = torch.tensor(encode(teach_text), dtype=torch.long)

    torch.save({"wash_train": wash_ids[:n_train],
                "wash_val": wash_ids[n_train:],
                "teach": teach_ids}, SCRATCH / "streams.pt")
    (RUN / "battery.json").write_text(json.dumps(
        {"probes": BATTERY, "banned": BANNED,
         "prompts": [prompt_of(p) for p in BATTERY],
         "sentences": [p["sentence"] for p in BATTERY]}, indent=2),
        encoding="utf-8")

    m = load_metrics()
    m["build"] = {
        "corpus": {"chars_total": len(_TEXT), "lines_total": len(lines),
                   "lines_dropped": len(dropped),
                   "wash_chars": len(wash_text), "wash_train_chars": n_train,
                   "vocab": 65},
        "teach": {"splice_gap": INSTALL["splice_gap"],
                  "n_statements_spliced": i, "teach_chars": len(teach_text)},
        "contamination": {
            "n_probes": len(scan),
            "all_subjects_absent": all(s["subject_absent"] for s in scan),
            "all_answers_absent": all(s["answer_absent"] for s in scan),
            "detail": scan},
        "battery_ns": {"fact": 20, "ctrl": 12, "near": 3, "tmpl": 19},
    }
    m["status"] = "BUILT (desk)"
    write_metrics(m)
    jlog("build_done", lines_dropped=len(dropped),
         wash_chars=len(wash_text), teach_chars=len(teach_text))


# ------------------------------------------------------------------ model
def make_model(device):
    from common import Cfg, TinyGPT
    model = TinyGPT(Cfg(**CFG)).to(device)
    n = sum(p.numel() for p in model.parameters())
    assert n == PARAMS_EXPECTED, n
    return model


def flat_grad(model, device):
    import torch
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                      if p.grad is not None])


def apply_flat(model, step) -> None:
    import torch
    i = 0
    for q in model.parameters():
        n = q.numel()
        q.data.add_(step[i:i + n].view_as(q.data))
        i += n


# ------------------------------------------------------------------ phase 2
def val_ce(model, val_ids, device, n_batches=12, bs=16, seed=VAL_SEED):
    import torch
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ctx = CFG["block_size"]
    tot = []
    for _ in range(n_batches):
        ix = torch.randint(len(val_ids) - ctx - 1, (bs,), generator=gen)
        x = torch.stack([val_ids[i:i + ctx] for i in ix]).to(device)
        y = torch.stack([val_ids[i + 1:i + 1 + ctx] for i in ix]).to(device)
        _, loss = model(x, y)
        tot.append(float(loss.item()))
    return sum(tot) / len(tot)


def cmd_train() -> None:
    import torch
    m = load_metrics()
    if m.get("organism", {}).get("done"):
        jlog("train_already_done")
        return
    while not thermal_gate("train"):
        time.sleep(60)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True     # training only
    torch.backends.cudnn.allow_tf32 = True

    ids = torch.tensor(encode(_TEXT), dtype=torch.long)
    n_train = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n_train], ids[n_train:]

    model = make_model(device)
    ck = CKPT / "e248_organism_train.pt"
    best = CKPT / "e248_organism_base.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                            weight_decay=0.1, betas=(0.9, 0.95))

    def lr_mult(s: int) -> float:
        w = TRAIN["warmup"]
        if s < w:
            return (s + 1) / w
        p = (s - w) / max(1, TRAIN["steps"] - w)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    step, hist, best_val = 0, [], float("inf")
    gen = torch.Generator().manual_seed(SEED_BASE)
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        gen.set_state(st["gen_state"])
        step, hist = st["step"], st.get("history", [])
        best_val = st.get("best_val", float("inf"))
        jlog("train_resume", step=step, best_val=best_val)

    m.setdefault("train_bursts", [])
    bursts = 0
    uturn_strikes = 0
    ctx = CFG["block_size"]
    while step < TRAIN["steps"] and bursts < TRAIN["max_bursts"]:
        while not thermal_gate(f"train-burst{bursts + 1}",
                               TRAIN["temp_start_c"]):
            time.sleep(60)
        t0 = time.time()
        stop = None
        burst_first_step = step
        model.train()
        while step < TRAIN["steps"]:
            step += 1
            ix = torch.randint(len(train_ids) - ctx - 1, (TRAIN["batch"],),
                               generator=gen)
            x = torch.stack([train_ids[i:i + ctx] for i in ix]).to(device)
            y = torch.stack([train_ids[i + 1:i + 1 + ctx] for i in ix]).to(device)
            if TRAIN["amp"]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            # per-step pacing (self-tuning duty, pynvml ~2 ms) + the
            # every-4-step break/hard guards
            pace_for_temp(ramp_in=(step - burst_first_step) < 25)
            if step % TRAIN["poll_every"] == 0:
                ok, why = mid_burst_check(step, f"train-burst{bursts + 1}", t0,
                                          TRAIN["burst_s"],
                                          break_c=TRAIN["temp_break_c"])
                if not ok:
                    stop = why
                    break
            if step % TRAIN["eval_every"] == 0 or step == TRAIN["steps"]:
                v = val_ce(model, val_ids, device)
                hist.append({"step": step,
                             "train_loss": float(loss.item()),
                             "val_loss": v, "t": now_iso()})
                if v < best_val - 1e-4:
                    best_val, uturn_strikes = v, 0
                    torch.save({"model": model.state_dict(), "step": step,
                                "val": v}, best)
                else:
                    uturn_strikes += 1
                jlog("train_eval", step=step, val=round(v, 4),
                     best=round(best_val, 4))
                if uturn_strikes >= 2 and v > best_val + TRAIN["uturn_tol"]:
                    stop = "uturn"
                    break
            if time.time() - t0 >= TRAIN["burst_s"]:
                stop = stop or "time"
                break
        bursts += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "gen_state": gen.get_state(),
                    "step": step, "history": hist, "best_val": best_val}, ck)
        m["train_bursts"].append({"burst": bursts, "end_step": step,
                                  "stop": stop, "best_val": best_val,
                                  "t": now_iso()})
        write_metrics(m)
        jlog("train_burst_end", burst=bursts, step=step, stop=stop,
             best_val=round(best_val, 4))
        if stop == "uturn":
            jlog("train_uturn_stop", note="the g1bS lesson honored")
            break
        if bursts < TRAIN["max_bursts"] and step < TRAIN["steps"]:
            if stop == "hard":
                # post-hard-stop (>= 82C read): heat-soak wait — >= 120 s
                # AND temp <= 65C (README rule; the 18:23 collision fix)
                t_wait = time.time()
                while (time.time() - t_wait < 120.0
                       or gpu_poll()["temp"] > 65.0):
                    time.sleep(20.0)
                    if time.time() - t_wait > 900.0:
                        jlog("heatsoak_timeout")
                        break
                jlog("heatsoak_done", temp=gpu_poll()["temp"])
            else:
                thermal_cooldown(TRAIN["cooldown_target_c"],
                                 TRAIN["cooldown_max_s"])

    stb = torch.load(best, map_location="cpu", weights_only=False)
    m["organism"] = {
        "params": PARAMS_EXPECTED, "cfg": CFG, "done": True,
        "bursts": bursts, "best_val": float(stb["val"]),
        "best_step": int(stb["step"]), "final_step": step,
        "gate_val_ce": TRAIN["val_ce_gate"],
        "G_ORGANISM_pass": bool(float(stb["val"]) <= TRAIN["val_ce_gate"]),
        "G_SIZE_pass": bool(PARAMS_EXPECTED <= 500_000_000),
        "size_reason": SIZE_REASON,
    }
    m["status"] = "ORGANISM-TRAINED"
    write_metrics(m)
    _train_png(hist)
    jlog("phase2_done", best_val=float(stb["val"]),
         gate=m["organism"]["G_ORGANISM_pass"])


def _train_png(hist) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot([h["step"] for h in hist], [h["val_loss"] for h in hist],
            "-o", ms=2.5, label="val", color="tab:blue")
    ax.plot([h["step"] for h in hist], [h["train_loss"] for h in hist],
            label="train (burst-last)", alpha=0.5, color="tab:orange")
    ax.axhline(TRAIN["val_ce_gate"], ls="--", c="r", label="gate 1.50")
    ax.axhline(TRAIN["val_ce_target"], ls=":", c="g", label="target 1.40")
    ax.set_xlabel("step")
    ax.set_ylabel("CE (nats/char)")
    ax.set_title(f"e248 the fresh organism ({PARAMS_EXPECTED:,} params, "
                 "12L/14H/896 char-LM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "e248_organism_training.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ battery read
def battery_read(model, device):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for p in BATTERY:
            ids = torch.tensor([prompt_ids(p)], dtype=torch.long,
                               device=device)
            logits, _ = model(ids)
            row = logits[0, -1, :].float()
            probs = torch.softmax(row, dim=-1)
            z = ans_index(p)
            pv = float(probs[z])
            top5 = torch.topk(probs, 5)
            top2v, top2i = torch.topk(probs, 2)
            sigma = float(row.std())
            out.append({
                "fact": p["fact"], "battery": p["battery"], "p": pv,
                "ans_char": p["ans_char"], "top1": ITOS[int(top2i[0])],
                "top2": ITOS[int(top2i[1])],
                "top1_p": float(top2v[0]),
                "ans_in_top5": bool(z in top5.indices.tolist()),
                "margin_sigma": float(top2v[0] - top2v[1]) / max(sigma, 1e-12),
                "argmax_is_answer": int(top2i[0]) == z,
                "logits": [float(x) for x in row.cpu()],
            })
    return out


def gate_pass(r: dict) -> bool:
    """e182's gate, char dialect: (top-1 and p>=0.8) or (top-5 and p>=0.5)."""
    return ((r["argmax_is_answer"] and r["p"] >= INSTALL["gate_top1"])
            or (r["ans_in_top5"] and r["p"] >= INSTALL["gate_top5"]))


def wash_ce(model, wash_val, device):
    return val_ce(model, wash_val, device, n_batches=12, bs=8)


# ------------------------------------------------------------------ phase 3
def cmd_install() -> None:
    import torch
    m = load_metrics()
    if m.get("install", {}).get("done"):
        jlog("install_already_done")
        return
    while not thermal_gate("install"):
        time.sleep(60)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    st = torch.load(SCRATCH / "streams.pt", weights_only=False)
    teach, wash_val = st["teach"], st["wash_val"]

    model = make_model(device)
    model.load_state_dict(torch.load(
        CKPT / "e248_organism_base.pt", map_location="cpu",
        weights_only=False)["model"])
    base_ce = wash_ce(model, wash_val, device)
    jlog("install_base_ce", base_ce=round(base_ce, 4))

    m.setdefault("install", {"rungs": [], "base_wash_ce": base_ce})
    rungs = m["install"]["rungs"]
    ck = CKPT / "e248_install_train.pt"
    inst = CKPT / "e248_installed_t0.pt"

    rung_list = ([(INSTALL["lr_ladder"][0], n) for n in INSTALL["passes_ladder"]]
                 + [(lr, n) for lr in INSTALL["lr_ladder"][1:]
                    for n in INSTALL["passes_at_high_lr"]])

    # resumable mid-rung state (optimizer moments included — a moments
    # reset would silently change the install trajectory)
    ri, pi, step, gen = 0, 0, 0, torch.Generator().manual_seed(SEED_INSTALL)
    opt = None
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        gen.set_state(s["gen_state"])
        ri, pi, step = s["ri"], s["pi"], s["step"]
        if s.get("opt") is not None:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-9,
                                    weight_decay=WASH["wd"],
                                    betas=WASH["betas"])
            opt.load_state_dict(s["opt"])
        jlog("install_resume", rung=ri, pass_i=pi, step=step,
             opt_state_restored=opt is not None)
    else:
        torch.save({"model": model.state_dict(),
                    "gen_state": gen.get_state(), "ri": 0, "pi": 0,
                    "step": 0}, ck)

    ctx = CFG["block_size"]
    accepted = None
    while ri < len(rung_list) and accepted is None:
        lr, npasses = rung_list[ri]
        while pi < npasses and accepted is None:
            if opt is None or opt.param_groups[0]["lr"] != lr:
                opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                        weight_decay=WASH["wd"],
                                        betas=WASH["betas"])
            while not thermal_gate(f"install-r{ri}-p{pi}"):
                time.sleep(60)
            t0 = time.time()
            model.train()
            target = (pi + 1) * 80
            while step < target:
                step += 1
                ix = torch.randint(len(teach) - ctx - 1, (WASH["batch"],),
                                   generator=gen)
                x = torch.stack([teach[i:i + ctx] for i in ix]).to(device)
                y = torch.stack([teach[i + 1:i + 1 + ctx] for i in ix]).to(device)
                _, loss = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step % WASH["poll_every"] == 0:
                    ok, why = mid_burst_check(step, f"install-r{ri}", t0,
                                              WASH["burst_s"])
                    if not ok:
                        jlog("install_burst_break", why=why, step=step)
                        break
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "gen_state": gen.get_state(), "ri": ri, "pi": pi,
                        "step": step}, ck)
            pi += 1
            # rung readout (pass end)
            reads = battery_read(model, device)
            ns, ps = {}, {}
            for r in reads:
                ns[r["battery"]] = ns.get(r["battery"], 0) + int(gate_pass(r))
                ps.setdefault(r["battery"], []).append(r["p"])
            meanp = sum(r["p"] for r in reads) / len(reads)
            ce = wash_ce(model, wash_val, device)
            g_inst = all(ns[b] >= INSTALL["floor"] for b in ns)
            p0_ok = INSTALL["p0_lo"] <= meanp <= INSTALL["p0_hi"]
            ce_ok = ce <= m["install"]["base_wash_ce"] * (1 + INSTALL["ce_wreck_tol"])
            rungs.append({"rung": ri, "lr": lr, "pass": pi,
                          "passes_target": npasses, "steps": step,
                          "gated_ns": ns, "mean_p0": meanp,
                          "wash_val_ce": ce, "G_INSTALL_pass": g_inst,
                          "p0_band_ok": p0_ok, "ce_ok": ce_ok,
                          "accepted": bool(g_inst and p0_ok and ce_ok),
                          "t": now_iso()})
            write_metrics(m)
            jlog("install_rung", rung=ri, lr=lr, pass_i=pi, gated=ns,
                 meanp=round(meanp, 3), ce=round(ce, 4))
            if rungs[-1]["accepted"]:
                accepted = (ri, lr, pi)
                break
            cooldown(30, 45)
        ri += 1
        pi = 0

    torch.save({"model": model.state_dict(), "rung": accepted,
                "gen_state": gen.get_state()}, inst)
    fin = battery_read(model, device)
    (RUN / "installed_t0_battery.json").write_text(json.dumps(fin, indent=2),
                                                   encoding="utf-8")
    ok = accepted is not None
    m["install"]["done"] = True
    m["install"]["accepted_rung"] = accepted
    m["install"]["G_INSTALL_final_pass"] = ok
    m["status"] = "INSTALLED" if ok else "INSTALL-FAILED"
    write_metrics(m)
    jlog("phase3_done", accepted=ok, rung=accepted)


# ------------------------------------------------------------------ phase 4
def _wash_grid(wname, mult=1):
    g = WASH["grids"][wname]
    return sorted({min(s * mult, 80 * mult) for s in g} | {80 * mult})


def cmd_wash() -> None:
    import numpy as np
    import torch
    m = load_metrics()
    m.setdefault("washes", {})
    mult = int(m.get("wash_ladder_mult", 1))
    while not thermal_gate("wash"):
        time.sleep(60)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    st = torch.load(SCRATCH / "streams.pt", weights_only=False)
    wash_train = st["wash_train"]
    ctx = CFG["block_size"]
    total = WASH["steps"] * mult
    model = make_model(device)

    for wname, seed in SEED_WASH.items():
        done_marker = RUN / f"wash_{wname}_m{mult}_done.json"
        if done_marker.exists():
            jlog(f"wash_{wname}_m{mult}_already_done")
            continue
        model.load_state_dict(torch.load(
            CKPT / "e248_installed_t0.pt", map_location="cpu",
            weights_only=False)["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=WASH["lr"],
                                weight_decay=WASH["wd"], betas=WASH["betas"])
        gen = torch.Generator().manual_seed(seed)
        grid = _wash_grid(wname, mult)
        gpath = SCRATCH / f"gdirs_{wname}_m{mult}.npy"
        gdir = np.lib.format.open_memmap(gpath, dtype=np.float16, mode="w+",
                                         shape=(total, PARAMS_EXPECTED))
        norms = []
        t0 = time.time()
        for t in range(1, total + 1):
            if t % 4 == 0 or t == 1:
                ok, why = mid_burst_check(t, f"wash-{wname}", t0,
                                          WASH["burst_s"])
                if not ok:
                    if why == "hard":
                        raise RuntimeError("thermal hard stop (>=82C)")
                    jlog("wash_pause", wash=wname, step=t, why=why,
                         temp=gpu_poll()["temp"])
                    time.sleep(45)
                    t0 = time.time()
            ix = torch.randint(len(wash_train) - ctx - 1, (WASH["batch"],),
                               generator=gen)
            x = torch.stack([wash_train[i:i + ctx] for i in ix]).to(device)
            y = torch.stack([wash_train[i + 1:i + 1 + ctx] for i in ix]).to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                flat = flat_grad(model, device)
                nrm = float(flat.norm().item())
                norms.append(nrm)
                gdir[t - 1] = (flat / max(nrm, 1e-12) * 16384.0).half().cpu().numpy()
            torch.nn.utils.clip_grad_norm_(model.parameters(), WASH["clip"])
            opt.step()
            if t in grid:
                torch.save({"model": model.state_dict(), "step": t,
                            "gen_state_after": gen.get_state(),
                            "batch_ce": float(loss.item())},
                           CKPT / f"e248_{wname}_m{mult}_s{t}.pt")
                jlog("wash_state_saved", wash=wname, step=t,
                     batch_ce=round(float(loss.item()), 4), gnorm=round(nrm, 3))
        gdir.flush()
        np.save(SCRATCH / f"gnorms_{wname}_m{mult}.npy", np.array(norms))
        done_marker.write_text(json.dumps(
            {"done": True, "steps": total, "seed": seed, "mult": mult,
             "grid": grid, "t": now_iso()}), encoding="utf-8")
        m["washes"][f"{wname}_m{mult}"] = {"seed": seed, "steps": total,
                                           "mult": mult, "grid": grid}
        write_metrics(m)
        jlog(f"wash_{wname}_m{mult}_done", steps=total)
        cooldown(45, 60)


def _load_state_sd(wname, s, mult=1):
    import torch
    if s == 0:
        return torch.load(CKPT / "e248_installed_t0.pt", map_location="cpu",
                          weights_only=False)["model"]
    return torch.load(CKPT / f"e248_{wname}_m{mult}_s{s}.pt",
                      map_location="cpu", weights_only=False)["model"]


def _dot_rows(A, B=None, device=None):
    """Pairwise cos dots of fp16 stored rows (unit*16384 convention):
    returns the (nA, nB) matrix of cosines, fp64, chunked. GPU path
    (fp32 matmul, fp64 accumulation) when device='cuda' is given."""
    import numpy as np
    nA = A.shape[0]
    nB = (B.shape[0] if B is not None else A.shape[0])
    if device == "cuda":
        import torch
        out = torch.zeros(nA, nB, dtype=torch.float64, device=device)
        col = 0
        while col < A.shape[1]:
            cend = min(col + CHUNK, A.shape[1])
            a = torch.from_numpy(np.ascontiguousarray(
                A[:, col:cend])).to(device, dtype=torch.float32)
            b = (torch.from_numpy(np.ascontiguousarray(
                B[:, col:cend])).to(device, dtype=torch.float32)
                if B is not None else a)
            out += (a @ b.T).double()
            col = cend
        return (out / (16384.0 ** 2)).cpu().numpy()
    out = np.zeros((nA, nB), dtype=np.float64)
    col = 0
    while col < A.shape[1]:
        cend = min(col + CHUNK, A.shape[1])
        a = A[:, col:cend].astype(np.float32)
        b = (B[:, col:cend].astype(np.float32) if B is not None else a)
        out += a.astype(np.float64) @ b.T.astype(np.float64)
        col = cend
    return out / (16384.0 ** 2)


def _row_norms(A):
    import numpy as np
    out = np.zeros(A.shape[0], dtype=np.float64)
    col = 0
    while col < A.shape[1]:
        cend = min(col + CHUNK, A.shape[1])
        out += (A[:, col:cend].astype(np.float64) ** 2).sum(axis=1)
        col = cend
    return np.sqrt(out) / 16384.0


def _gstate_path(wname, s):
    return SCRATCH / f"gstate_{wname}_{s}.npy"


def _sup_cache_path():
    return SCRATCH / "supports_t0.npy"


def _sup_state_path(wname, s):
    return SCRATCH / f"supports_{wname}_s{s}.npy"


def cmd_read() -> None:
    import numpy as np
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    m = load_metrics()
    mult = int(m.get("wash_ladder_mult", 1))
    m.setdefault("reads", {})
    if m["reads"].get("_mult") not in (None, mult):
        # the ladder moved: all mult-dependent reads are stale; recompute
        m["reads"] = {"_mult_reset_from": m["reads"]["_mult"]}
        write_metrics(m)
        jlog("reads_reset_for_mult", mult=mult)
    m["reads"]["_mult"] = mult
    st = torch.load(SCRATCH / "streams.pt", weights_only=False)
    wash_train = st["wash_train"]
    ctx = CFG["block_size"]
    model = make_model(device)

    # (A) journals ------------------------------------------------------
    jp = RUN / "wash_journal.json"
    journal = json.loads(jp.read_text()) if jp.exists() else {}
    states_all = {"t0": [0],
                  "w1": _wash_grid("w1", mult), "w2": _wash_grid("w2", mult)}
    for wname, states in states_all.items():
        for s in states:
            key = (f"{wname}:{s}" if wname == "t0"
                   else f"{wname}_m{mult}:{s}")
            if key in journal:
                continue
            if wname == "t0":
                sd = _load_state_sd(None, 0)
            else:
                p = CKPT / f"e248_{wname}_m{mult}_s{s}.pt"
                if not p.exists():
                    jlog("journal_state_missing", key=key)
                    continue
                sd = _load_state_sd(wname, s, mult)
            model.load_state_dict(sd)
            journal[key] = battery_read(model, device)
            jp.write_text(json.dumps(journal), encoding="utf-8")
            jlog("journal_state", key=key)
    # G_REPRO: t0 re-read
    model.load_state_dict(_load_state_sd(None, 0))
    r2 = battery_read(model, device)
    max_dp = max(abs(a["p"] - b["p"]) for a, b in zip(journal["t0:0"], r2))
    m["reads"]["G_REPRO"] = {"max_dp_t0_reread": max_dp,
                             "pass": bool(max_dp <= 0.005)}
    # G_RECALIBRATION [amendment (i)]: the census thresholds' mass check
    if "G_RECALIBRATION" not in m["reads"]:
        j0 = journal["t0:0"]
        margins0 = [r["margin_sigma"] for r in j0]
        ps0 = [r["p"] for r in j0]
        med_m = float(np.median(margins0))
        p25_m = float(np.percentile(margins0, 25))
        med_p = float(np.median(ps0))
        passed = (med_m >= BARS["recal_margin_med_min"]
                  and p25_m >= BARS["recal_margin_p25_min"]
                  and BARS["recal_p_lo"] <= med_p <= BARS["recal_p_hi"])
        if passed:
            fz = 0.05
        else:
            fz = 0.05 * med_m / BARS["committed_anchor_margin_median"]
        m["reads"]["G_RECALIBRATION"] = {
            "median_t0_margin": med_m, "p25_t0_margin": p25_m,
            "median_t0_p": med_p, "pass": bool(passed),
            "census_flip_zone": fz,
            "note": ("verbatim thresholds ported" if passed else
                     f"RECALIBRATED flip zone {fz:.4f} (disclosed; the "
                     "verbatim-zone census co-reports at adjudication)")}
        write_metrics(m)
        jlog("G_RECALIBRATION", pass_=passed, med_m=round(med_m, 4),
             p25_m=round(p25_m, 4), med_p=round(med_p, 4), fz=round(fz, 5))
    write_metrics(m)
    jlog("G_REPRO", max_dp=max_dp)

    # (B) t0 supports ----------------------------------------------------
    sup_path = _sup_cache_path()
    if not sup_path.exists():
        gdir = np.lib.format.open_memmap(sup_path, dtype=np.float16,
                                         mode="w+",
                                         shape=(54, PARAMS_EXPECTED))
        model.load_state_dict(_load_state_sd(None, 0))
        for i, p in enumerate(BATTERY):
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([prompt_ids(p)], dtype=torch.long,
                               device=device)
            probs = _probe_probs(model, ids)
            loss = -torch.log(probs[ans_index(p)] + 1e-12)
            loss.backward()
            flat = flat_grad(model, device).detach()
            nrm = float(flat.norm().item())
            gdir[i] = (flat / max(nrm, 1e-12) * 16384.0).half().cpu().numpy()
            if i % 9 == 0:
                jlog("supports_t0", i=i)
        gdir.flush()
    sup = np.memmap(sup_path, dtype=np.float16, mode="r",
                    shape=(54, PARAMS_EXPECTED))
    # G_SUPPORTFD: p(theta + 0.02 s_hat) > p0
    if "G_SUPPORTFD" not in m["reads"]:
        model.load_state_dict(_load_state_sd(None, 0))
        p0s = [r["p"] for r in journal["t0:0"]]
        fd = []
        for i, p in enumerate(BATTERY):
            s_hat = torch.from_numpy(sup[i].astype(np.float32)).to(device)
            s_hat = s_hat / (s_hat.norm() + 1e-12)
            apply_flat(model, s_hat * 0.02)
            ids = torch.tensor([prompt_ids(p)], dtype=torch.long,
                               device=device)
            with torch.no_grad():
                pv = float(_probe_probs(model, ids)[ans_index(p)])
            apply_flat(model, -(s_hat * 0.02))
            fd.append(pv - p0s[i])
        m["reads"]["G_SUPPORTFD"] = {
            "n": len(fd), "n_fail": sum(1 for d in fd if d <= 0),
            "min_dp": float(min(fd)),
            "pass": bool(all(d > 0 for d in fd))}
        write_metrics(m)
        jlog("G_SUPPORTFD", n_fail=m["reads"]["G_SUPPORTFD"]["n_fail"])

    # (C) wash gradient states -------------------------------------------
    for wname in ("w1", "w2"):
        for s in ([0] + _wash_grid(wname, mult)):
            gp = _gstate_path(wname, s)
            if gp.exists():
                continue
            model.load_state_dict(_load_state_sd(
                None if s == 0 else wname, s, mult))
            gen = torch.Generator().manual_seed(SEED_WASH[wname])
            for _ in range(s + 1):        # draw #(s+1): the state's OWN next
                ix = torch.randint(len(wash_train) - ctx - 1,
                                   (WASH["batch"],), generator=gen)
            x = torch.stack([wash_train[i:i + ctx] for i in ix]).to(device)
            y = torch.stack([wash_train[i + 1:i + 1 + ctx] for i in ix]).to(device)
            _, loss = model(x, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                flat = flat_grad(model, device)
                nrm = float(flat.norm().item())
                np.save(gp, (flat / max(nrm, 1e-12) * 16384.0).half().cpu().numpy())
            jlog("wash_grad_state", key=f"{wname}:{s}", gnorm=round(nrm, 3))
    # G_WASHGRADFD: CE(W - 0.002 g_hat) < CE(W) at s=0 and the deepest state
    if "G_WASHGRADFD" not in m["reads"]:
        wfd = []
        for wname in ("w1", "w2"):
            for s in ([0, _wash_grid(wname, mult)[-1]]):
                model.load_state_dict(_load_state_sd(
                    None if s == 0 else wname, s, mult))
                gen = torch.Generator().manual_seed(SEED_WASH[wname])
                for _ in range(s + 1):
                    ix = torch.randint(len(wash_train) - ctx - 1,
                                       (WASH["batch"],), generator=gen)
                x = torch.stack([wash_train[i:i + ctx] for i in ix]).to(device)
                y = torch.stack([wash_train[i + 1:i + 1 + ctx] for i in ix]).to(device)
                _, loss0 = model(x, y)
                g_hat = torch.from_numpy(
                    np.load(_gstate_path(wname, s)).astype(np.float32)).to(device)
                g_hat = g_hat / (g_hat.norm() + 1e-12)
                with torch.no_grad():
                    apply_flat(model, -(g_hat * 0.002))
                    _, loss1 = model(x, y)
                    apply_flat(model, g_hat * 0.002)
                wfd.append(float(loss1.item() - loss0.item()))
        m["reads"]["G_WASHGRADFD"] = {"dces": wfd,
                                      "pass": bool(all(d < 0 for d in wfd))}
        write_metrics(m)
        jlog("G_WASHGRADFD", pass_=m["reads"]["G_WASHGRADFD"]["pass"])

    # (D) the Grams + span bases ------------------------------------------
    spans = {}
    for wname in ("w1", "w2"):
        Gp = SCRATCH / f"gram_{wname}_m{mult}.npy"
        if Gp.exists():
            G = np.load(Gp)
        else:
            gdir = np.memmap(SCRATCH / f"gdirs_{wname}_m{mult}.npy",
                             dtype=np.float16, mode="r",
                             shape=(WASH["steps"] * mult, PARAMS_EXPECTED))
            G = _dot_rows(gdir, device=("cuda" if device == "cuda" else None))
            np.save(Gp, G)
            del gdir
        H = min(40 * mult, G.shape[0] - 1)
        A = G[:H, :H].astype(np.float64)
        w_eig, V = np.linalg.eigh(A)
        order = np.argsort(-w_eig)
        Vk = V[:, order[:2]]                     # (H, 2)
        spans[wname] = {"eigvals": [float(w_eig[order[0]]),
                                    float(w_eig[order[1]])],
                        "Vk": Vk, "gram": G, "half": H}
        jlog("span_built", wash=wname, top_eig=float(w_eig[order[0]]))
    # basis vectors materialized (rows @ Vk, renormalized)
    basis = {}
    for wname in ("w1", "w2"):
        H = spans[wname]["half"]
        Vk = spans[wname]["Vk"]
        bpath = SCRATCH / f"basis_{wname}_m{mult}.npy"
        if bpath.exists():
            basis[wname] = np.load(bpath)
            continue
        gdir = np.memmap(SCRATCH / f"gdirs_{wname}_m{mult}.npy",
                         dtype=np.float16, mode="r",
                         shape=(WASH["steps"] * mult, PARAMS_EXPECTED))
        Bs = np.lib.format.open_memmap(bpath, dtype=np.float32, mode="w+",
                                       shape=(2, PARAMS_EXPECTED))
        for k in range(2):
            acc = np.zeros(PARAMS_EXPECTED, dtype=np.float64)
            col = 0
            while col < PARAMS_EXPECTED:
                cend = min(col + CHUNK, PARAMS_EXPECTED)
                blk = gdir[:H, col:cend].astype(np.float32)
                acc[col:cend] = (blk.T.astype(np.float64) @ Vk[:, k])
                col = cend
            nrm = np.linalg.norm(acc)
            Bs[k] = (acc / max(nrm, 1e-12)).astype(np.float32)
        Bs.flush()
        basis[wname] = np.load(bpath)
        del gdir
    ident = float(abs(np.dot(basis["w1"][0], basis["w2"][0])))
    second = float(abs(np.dot(basis["w1"][1], basis["w2"][1])))
    m["reads"]["span"] = {
        "mult": mult, "top_identity": ident, "second_dim_cos": second,
        "eigvals": {w: spans[w]["eigvals"] for w in spans}}
    write_metrics(m)
    jlog("span_identity", top=round(ident, 4), second=round(second, 4))

    # wind_cums: steps H+1..end decomposed; cos(P_span g_t, s_i(0))
    # b_k = sum_j Vk[j,k] u_j normalized -> nrm_k = sqrt(Vk[:,k]^T G Vk[:,k]);
    # g_t.b_k = (G[t,:H] @ Vk[:,k]) / nrm_k  (the desk-review bug fix:
    # without the /nrm_k the projected-cosine is skewed by diag(nrm))
    if "wind_cum" not in m["reads"]:
        wind = {}
        for wname in ("w1", "w2"):
            G = spans[wname]["gram"]
            H = spans[wname]["half"]
            Vk = spans[wname]["Vk"]
            nrm_k = np.sqrt(np.maximum(
                [float(Vk[:, k] @ (G[:H, :H] @ Vk[:, k])) for k in range(2)],
                1e-24))
            gb = np.stack([(G[H:, :H] @ Vk[:, k]) / nrm_k[k]
                           for k in range(2)])                 # (2, T2)
            pg_norm = np.sqrt((gb ** 2).sum(axis=0)) + 1e-12
            bs = np.zeros((2, 54), dtype=np.float64)     # b_k . s_i
            col = 0
            while col < PARAMS_EXPECTED:
                cend = min(col + CHUNK, PARAMS_EXPECTED)
                Sb = sup[:, col:cend].astype(np.float64)
                for k in range(2):
                    bs[k] += Sb @ basis[wname][k][col:cend].astype(np.float64)
                col = cend
            bs /= 16384.0        # sup rows are unit*16384 (fp16 cache)
            proj = (gb[:, None, :] * bs[:, :, None]).sum(axis=0)   # (T2, 54)
            cosmat = proj / (pg_norm[:, None] * 1.0)
            wind[wname] = np.abs(cosmat).sum(axis=0).tolist()
        m["reads"]["wind_cum"] = wind
        write_metrics(m)
        jlog("wind_cum_done")

    # (E) the seat alignment curves ----------------------------------------
    if "seat_align" not in m["reads"]:
        seat = {}
        supnorm = _row_norms(sup)
        for wname in ("w1", "w2"):
            for s in ([0] + _wash_grid(wname, mult)):
                g = np.load(_gstate_path(wname, s)).astype(np.float64)
                g = g / (np.linalg.norm(g) + 1e-12)
                dots = np.zeros(54)
                col = 0
                while col < PARAMS_EXPECTED:
                    cend = min(col + CHUNK, PARAMS_EXPECTED)
                    dots += sup[:, col:cend].astype(np.float64) @ g[col:cend]
                    col = cend
                seat[f"{wname}:{s}"] = (dots / 16384.0 / supnorm).tolist()
                jlog("seat_read", key=f"{wname}:{s}")
        m["reads"]["seat_align"] = seat
        write_metrics(m)

    # (F) supports at the deep states --------------------------------------
    deep_states = sorted({min(50 * mult, WASH["steps"] * mult),
                          min(80 * mult, WASH["steps"] * mult)})
    for wname in ("w1", "w2"):
        for s in deep_states:
            path = _sup_state_path(wname, s)
            if path.exists():
                continue
            model.load_state_dict(_load_state_sd(wname, s, mult))
            mm = np.lib.format.open_memmap(path, dtype=np.float16, mode="w+",
                                           shape=(54, PARAMS_EXPECTED))
            for i, p in enumerate(BATTERY):
                model.zero_grad(set_to_none=True)
                ids = torch.tensor([prompt_ids(p)], dtype=torch.long,
                                   device=device)
                pr = _probe_probs(model, ids)
                loss = -torch.log(pr[ans_index(p)] + 1e-12)
                loss.backward()
                flat = flat_grad(model, device).detach()
                nrm = float(flat.norm().item())
                mm[i] = (flat / max(nrm, 1e-12) * 16384.0).half().cpu().numpy()
                if i % 18 == 0:
                    jlog("supports_state", wash=wname, s=s, i=i)
            mm.flush()
    if "coherence" not in m["reads"]:
        coh, rot, cross = {}, {w: {} for w in ("w1", "w2")}, {}
        dev = ("cuda" if device == "cuda" else None)
        for s in deep_states:
            A = np.memmap(_sup_state_path("w1", s), dtype=np.float16,
                          mode="r", shape=(54, PARAMS_EXPECTED))
            B = np.memmap(_sup_state_path("w2", s), dtype=np.float16,
                          mode="r", shape=(54, PARAMS_EXPECTED))
            c10 = _dot_rows(A, sup, device=dev)
            c20 = _dot_rows(B, sup, device=dev)
            c12 = _dot_rows(A, B, device=dev)
            t1 = np.arccos(np.clip(c10.diagonal(), -1, 1))
            t2 = np.arccos(np.clip(c20.diagonal(), -1, 1))
            tB = np.arccos(np.clip(c12.diagonal(), -1, 1))
            ratio = tB / np.maximum(t1 + t2, 1e-12)
            coh[str(s)] = {"median_ratio": float(np.median(ratio)),
                           "coherence": float(1 - np.median(ratio))}
            rot["w1"][str(s)] = c10.diagonal().tolist()
            rot["w2"][str(s)] = c20.diagonal().tolist()
            cross[str(s)] = c12.diagonal().tolist()
        m["reads"]["coherence"] = coh
        m["reads"]["rotation"] = rot
        m["reads"]["crosswash_cos"] = cross
        write_metrics(m)
        jlog("coherence_done", coh=coh)

    # (G) thermal fits -------------------------------------------------------
    if "thermal" not in m["reads"]:
        m["reads"]["thermal"] = thermal_fits(journal, mult)
        write_metrics(m)
        jlog("thermal_fits_done")

    # the wash-depth ladder decision (pre-registered): if the fact battery
    # declined < DEPTH_SOFT at the deepest state, queue the next rung
    wh = m.get("washes", {})
    if not wh.get("ladder_decided"):
        p0j = {r["fact"]: r["p"] for r in journal["t0:0"]}
        decls = {}
        for wname in ("w1", "w2"):
            deepest = _wash_grid(wname, mult)[-1]
            key = f"{wname}_m{mult}:{deepest}"
            if key in journal:
                rows = [r for r in journal[key] if r["battery"] == "fact"]
                decls[wname] = float(np.mean(
                    [1 - r["p"] / max(p0j[r["fact"]], 1e-9) for r in rows]))
        wh["fact_decl"] = decls
        min_decl = min(decls.values()) if decls else 1.0
        if min_decl < DEPTH_SOFT and mult < WASH_LADDER[-1]:
            nxt = [x for x in WASH_LADDER if x > mult][0]
            m["wash_ladder_mult"] = nxt
            wh["ladder_decided"] = True
            wh["ladder_next"] = nxt
            jlog("wash_ladder_next", from_mult=mult, to_mult=nxt,
                 fact_decl=decls)
        else:
            wh["ladder_decided"] = True
            wh["ladder_next"] = None
            jlog("wash_ladder_final", mult=mult, fact_decl=decls)
        m["washes"] = wh
        write_metrics(m)

    m["status"] = "READ (phase 4b done)"
    jlog("phase4b_done")


def _probe_probs(model, ids):
    import torch
    logits, _ = model(ids)
    return torch.softmax(logits[0, -1, :].float(), dim=-1)


def thermal_fits(journal, mult=1) -> dict:
    import numpy as np
    from scipy.stats import spearmanr
    j0 = journal["t0:0"]
    L0 = np.array([r["logits"] for r in j0])
    p0 = np.array([r["p"] for r in j0])
    Z = np.array([ans_index(p) for p in BATTERY])

    def q_of_T(T):
        zs = L0 / T
        zs = zs - zs.max(axis=1, keepdims=True)
        e = np.exp(zs)
        pr = e / e.sum(axis=1, keepdims=True)
        return pr[np.arange(len(Z)), Z]

    def nll(T, obs):
        q = np.clip(q_of_T(T), 1e-12, 1 - 1e-12)
        return float(-(obs * np.log(q) + (1 - obs) * np.log(1 - q)).sum())

    out = {"cells": [], "resid_join": []}
    for wname in ("w1", "w2"):
        wkeys = sorted((k for k in journal if k.startswith(f"{wname}_m")),
                       key=lambda k: int(k.split(":")[1]))
        states = [int(k.split(":")[1]) for k in wkeys]
        # deep states = the two largest journal states of this wash
        deep_states = set(states[-2:])
        for key, s in zip(wkeys, states):
            obs = np.array([r["p"] for r in journal[key]])
            Ts = np.linspace(1.0, 12.0, 441)
            vals = [nll(T, obs) for T in Ts]
            Tstar = float(Ts[int(np.argmin(vals))])
            lo, hi = max(1.0, Tstar - 0.03), Tstar + 0.03
            for _ in range(60):
                a = lo + 0.382 * (hi - lo)
                b = lo + 0.618 * (hi - lo)
                if nll(a, obs) < nll(b, obs):
                    hi = b
                else:
                    lo = a
            Tstar = float(0.5 * (lo + hi))
            qs = np.clip(q_of_T(Tstar), 1e-12, 1 - 1e-12)
            ss_res = float(((obs - qs) ** 2).sum())
            ss_tot = float(((obs - p0) ** 2).sum())
            out["cells"].append({"wash": wname, "state": s, "T": Tstar,
                                 "R2_pooled": 1 - ss_res / max(ss_tot, 1e-12),
                                 "mean_p_obs": float(obs.mean()),
                                 "mean_p0": float(p0.mean())})
            if s in deep_states:
                resid = obs - qs
                eros = 1 - obs / np.maximum(p0, 1e-12)
                bidx = {b: [i for i, p in enumerate(BATTERY)
                            if p["battery"] == b] for b in
                        ("fact", "ctrl", "tmpl")}

                def ranks(v):
                    return (np.argsort(np.argsort(v)) + 1) / len(v)

                xs, ys = [], []
                for b in ("fact", "ctrl", "tmpl"):
                    ii = bidx[b]
                    xs += list(ranks(-resid[ii]))
                    ys += list(ranks(-eros[ii]))
                out["resid_join"].append(
                    {"wash": wname, "state": s,
                     "rho_blocked": float(spearmanr(xs, ys).statistic),
                     "mean_abs_resid": float(np.abs(resid).mean()),
                     "n": len(xs)})
    return out


# ------------------------------------------------------------------ phase 5
def cmd_adjudicate() -> None:
    import numpy as np
    m = load_metrics()
    mult = int(m.get("wash_ladder_mult", 1))
    jr = json.loads((RUN / "wash_journal.json").read_text(encoding="utf-8"))
    reads = m.get("reads", {})
    grids = {"w1": _wash_grid("w1", mult), "w2": _wash_grid("w2", mult)}

    gates = {
        "G_ORGANISM": bool(m.get("organism", {}).get("G_ORGANISM_pass")),
        "G_SIZE": bool(m.get("organism", {}).get("G_SIZE_pass")),
        "G_INSTALL": bool(m.get("install", {}).get("G_INSTALL_final_pass")),
        "G_CORPUS": bool(m.get("build", {}).get("contamination", {})
                         .get("all_subjects_absent") and
                         m.get("build", {}).get("contamination", {})
                         .get("all_answers_absent")),
        "G_REPRO": bool(reads.get("G_REPRO", {}).get("pass")),
        "G_SUPPORTFD": bool(reads.get("G_SUPPORTFD", {}).get("pass")),
        "G_WASHGRADFD": bool(reads.get("G_WASHGRADFD", {}).get("pass")),
    }
    # non-kill gate [amendment (i)]: a fail recalibrates + discloses, it
    # does not kill the cell
    recal = reads.get("G_RECALIBRATION", {})
    gates["G_RECALIBRATION_nonkill"] = {
        "pass": bool(recal.get("pass")), "flip_zone": recal.get("census_flip_zone")}
    all_gates = all(v for k, v in gates.items()
                    if isinstance(v, bool))
    bars_out = {}

    # ---- bar 1 + 5: the census -----------------------------------------
    fz = reads.get("G_RECALIBRATION", {}).get("census_flip_zone", 0.05)
    fz_verbatim = 0.05
    census = {}
    census_verbatim = {}
    for zone, fz_use, sink in ((fz, fz, census), (fz_verbatim, fz_verbatim, census_verbatim)):
        for b in ("fact", "ctrl", "tmpl", "near"):
            for w in ("w1", "w2"):
                states = [0] + grids[w]
                rows = {s: {r["fact"]: r for r in
                            (jr["t0:0"] if s == 0
                             else jr.get(f"{w}_m{mult}:{s}", []))}
                        for s in states}
                b_facts = {p["fact"] for p in BATTERY if p["battery"] == b}
                classes = {}
                for fact in rows[0]:
                    if fact not in b_facts:
                        continue
                    p0 = rows[0][fact]["p"]
                    p_seq = [rows[s][fact]["p"] for s in states if s in rows
                             and fact in rows[s]]
                    m_seq = [rows[s][fact]["margin_sigma"] for s in states
                             if s in rows and fact in rows[s]]
                    p_cross = next((k for k in range(1, len(p_seq))
                                    if p_seq[k] < 0.5 * p0), None)
                    m_cross = next((k for k in range(1, len(m_seq))
                                    if m_seq[k] < fz_use), None)
                    if p_cross is None and m_cross is None:
                        cls = "NEITHER"
                    elif m_cross is None:
                        cls = "P-FIRST"
                    elif p_cross is None:
                        cls = "MARGIN-FIRST"
                    elif p_cross == m_cross:
                        cls = "TOGETHER"
                    elif p_cross < m_cross:
                        cls = "P-FIRST"
                    else:
                        cls = "MARGIN-FIRST"
                    classes[fact] = {"class": cls, "p0": p0,
                                     "p_last": p_seq[-1], "m_last": m_seq[-1]}
                counts = {}
                for c in classes.values():
                    counts[c["class"]] = counts.get(c["class"], 0) + 1
                sink[f"{b}-{w}"] = {"counts": counts, "n": len(classes),
                                    "classes": classes}
    cells_adj = [f"{b}-{w}" for b in ("fact", "ctrl", "tmpl")
                 for w in ("w1", "w2")]
    pfirst_cells = sum(
        1 for c in cells_adj
        if census[c]["counts"].get("P-FIRST", 0) >
        max(census[c]["counts"].get("MARGIN-FIRST", 0),
            census[c]["counts"].get("TOGETHER", 0)))
    bars_out["1_PFIRST_MODALITY"] = {
        "cells_modal": pfirst_cells, "need": BARS["pfirst_cells_need"],
        "pass": pfirst_cells >= BARS["pfirst_cells_need"],
        "census_flip_zone_used": fz,
        "census_counts": {c: census[c]["counts"] for c in census},
        "verbatim_zone_coreport": ({c: census_verbatim[c]["counts"]
                                    for c in census_verbatim}
                                   if fz != 0.05 else "identical (zone ported verbatim)")}

    zomb, zomb_ok, stand_modal = {}, 0, 0
    for c in cells_adj:
        pf = [f for f, d in census[c]["classes"].items()
              if d["class"] == "P-FIRST"]
        n = len(pf)
        standing = sum(
            1 for f in pf
            if census[c]["classes"][f]["m_last"] >= fz
            and census[c]["classes"][f]["p_last"] < 0.5 *
            census[c]["classes"][f]["p0"])
        frac = standing / n if n else float("nan")
        committed = COMMITTED["zombie_standing"][c]
        lo = committed / BARS["zomb_factor"]
        hi = min(1.0, committed * BARS["zomb_factor"])
        ok = (frac == frac) and lo - 1e-9 <= frac <= hi + 1e-9
        zomb[c] = {"n_pfirst": n, "standing": standing, "frac": frac,
                   "committed": committed, "band": [lo, hi], "in_band": ok}
        zomb_ok += int(ok)
        stand_modal += int(frac == frac and frac >= BARS["zomb_stand_line"])
    bars_out["5_ZOMBIE_RATES"] = {
        "cells_in_band": zomb_ok, "cells_standing": stand_modal,
        "need": BARS["zomb_cells_need"],
        "pass": zomb_ok >= BARS["zomb_cells_need"]
        and stand_modal >= BARS["zomb_cells_need"], "detail": zomb}

    # ---- bar 2: the seat -------------------------------------------------
    seat_out, z_fire, dir_fire, nosplit = {}, 0, 0, False
    for w in ("w1", "w2"):
        rows80 = jr[f"{w}_m{mult}:{grids[w][-1]}"]
        fam = [i for i, p in enumerate(BATTERY) if p["family"] == "product"]
        hrs = {i: rows80[i]["p"] / max(jr["t0:0"][i]["p"], 1e-12)
               for i in fam}
        dying = min(hrs, key=hrs.get)
        holder = max(hrs, key=hrs.get)
        split = hrs[holder] / max(hrs[dying], 1e-12)
        band = [i for i in fam if i not in (dying, holder)]
        w_reads = {}
        for s in ([0] + grids[w]):
            A = reads.get("seat_align", {}).get(f"{w}:{s}")
            if A is None:
                continue
            A = np.array(A)
            sig = float(np.std([A[i] for i in band], ddof=1))
            w_reads[str(s)] = {
                "A_holder": float(A[holder]), "A_dying": float(A[dying]),
                "sigma": sig,
                "Z": float(abs(A[holder] - A[dying]) / max(sig, 1e-12)),
                "ratio_abs": float(abs(A[dying]) /
                                   max(abs(A[holder]), 1e-12))}
        seat_out[w] = {"dying": BATTERY[dying]["fact"],
                       "holder": BATTERY[holder]["fact"],
                       "fate_split": float(split), "reads": w_reads}
        if split >= BARS["seat_fate_split"]:
            z_fire += int(any(r["Z"] >= BARS["seat_z_line"]
                              for r in w_reads.values()))
            dir_fire += int(sum(
                1 for s, r in w_reads.items()
                if int(s) in (10 * mult, 50 * mult, 80 * mult)
                and r["ratio_abs"] >= BARS["seat_ratio_line"]) >= 1)
        else:
            nosplit = True
    bars_out["2_SEAT_Z_DIRECTION"] = {
        "z_fire_washes": z_fire, "dir_fire_washes": dir_fire,
        "nosplit_wash_seen": nosplit,
        "pass": z_fire >= 1 and dir_fire >= 1, "detail": seat_out}

    # ---- bar 3: T-R2 ------------------------------------------------------
    cells = reads.get("thermal", {}).get("cells", [])
    deep_set = {50 * mult, 80 * mult}
    tr2_detail = []
    for c in cells:
        if c["state"] in deep_set:
            tr2_detail.append({
                "cell": f"{c['wash']}-{c['state']}", "R2": c["R2_pooled"],
                "T": c["T"],
                "in_band": BARS["tr2_lo"] <= c["R2_pooled"] <= BARS["tr2_hi"]})
    washes_seen = {d["cell"].split("-")[0] for d in tr2_detail if d["in_band"]}
    n_in_band = sum(1 for d in tr2_detail if d["in_band"])
    bars_out["3_T_R2"] = {
        "detail": tr2_detail, "cells_in_band": n_in_band,
        "washes_represented": sorted(washes_seen),
        "need": BARS["tr2_cells_need"],
        "pass": n_in_band >= 2 and len(washes_seen) == 2}

    # ---- bar 4: span identity ---------------------------------------------
    ident = reads.get("span", {}).get("top_identity")
    bars_out["4_SPAN_IDENTITY"] = {
        "identity": ident, "line": BARS["span_identity_line"],
        "pass": bool(ident is not None
                     and ident > BARS["span_identity_line"])}

    # ---- co-bars -----------------------------------------------------------
    coh = reads.get("coherence", {})
    deep_keys = [str(grids["w1"][-2]), str(grids["w1"][-1])]
    bars_out["6_COMMON_MODE"] = {
        "coherence": {k: coh.get(k, {}).get("coherence") for k in deep_keys},
        "pass": all(coh.get(k, {}).get("coherence", 0)
                    >= BARS["coherence_line"] for k in deep_keys)}
    wind = reads.get("wind_cum", {})
    exp_detail, exp_ok = {}, 0
    for w in ("w1", "w2"):
        if w not in wind or w not in seat_out:
            continue
        dying_fact = seat_out[w]["dying"]
        holder_fact = seat_out[w]["holder"]
        di = next(i for i, p in enumerate(BATTERY) if p["fact"] == dying_fact)
        hi_ = next(i for i, p in enumerate(BATTERY) if p["fact"] == holder_fact)
        ratio = wind[w][di] / max(wind[w][hi_], 1e-12)
        exp_detail[w] = float(ratio)
        exp_ok += int(BARS["exposure_lo"] <= ratio <= BARS["exposure_hi"])
    bars_out["7_ANCHOR_EXPOSURE"] = {
        "ratios": exp_detail, "band": [BARS["exposure_lo"], BARS["exposure_hi"]],
        "pass": exp_ok >= BARS["exposure_washes_need"]}
    rj = reads.get("thermal", {}).get("resid_join", [])
    rj_ok = [r for r in rj
             if abs(r["rho_blocked"]) >= BARS["resid_rho_line"]
             and r.get("mean_abs_resid", 0.0) > BARS["resid_mag_floor"]]
    bars_out["8_RESIDUAL_ORDER"] = {
        "detail": rj, "n_ok": len(rj_ok),
        "rho_line": BARS["resid_rho_line"],
        "mag_floor": BARS["resid_mag_floor"],
        "amendment_note": ("phase-boundary amendment (pre-wash-compute, "
                           "coordinator advisory): both clauses per cell — "
                           "|rho| >= 0.85 (above the measured seed-0 "
                           "pure-temperature null max 0.812) AND "
                           "mean_abs_resid > 10x the null's magnitude"),
        "pass": len(rj_ok) >= BARS["resid_cells_need"]}

    primaries = ["1_PFIRST_MODALITY", "2_SEAT_Z_DIRECTION", "3_T_R2",
                 "4_SPAN_IDENTITY", "5_ZOMBIE_RATES"]
    n_pass = sum(1 for b in primaries if bars_out[b]["pass"])
    if not all_gates:
        verdict = "INSTRUMENT-DEAD"
    elif n_pass == 5:
        verdict = "REPLICATES"
    else:
        verdict = "PARTIAL"

    m["adjudication"] = {
        "gates": gates, "all_gates_pass": all_gates, "bars": bars_out,
        "primaries_passed": n_pass, "verdict": verdict,
        "clause": (f"{n_pass}/5 primaries within frozen tolerance; "
                   f"co-bars "
                   + json.dumps({k: bars_out[k]["pass"] for k in
                                 ("6_COMMON_MODE", "7_ANCHOR_EXPOSURE",
                                  "8_RESIDUAL_ORDER")})
                   + "; gates "
                   + ("ALL PASS" if all_gates else
                      "FAILED: " + str([k for k, v in gates.items()
                                        if not v]))
                   + "; span identity carries the sign-shadow guard "
                   "(recipe+organism at minimum; e234-validated on "
                   "GPT-2's washes only; corpus-swap named, not run)"),
        "committed_re_read": COMMITTED, "wash_ladder_mult": mult,
    }
    write_metrics(m)
    _adjudication_png(census, zomb, bars_out, verdict)
    (RUN / "NOTES_entry_draft.md").write_text(
        _notes_draft(verdict, n_pass, bars_out, gates), encoding="utf-8")
    jlog("phase5_verdict", verdict=verdict, primaries=n_pass)


def _adjudication_png(census, zomb, bars, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    cells = [f"{b}-{w}" for b in ("fact", "ctrl", "tmpl") for w in ("w1", "w2")]
    x = np.arange(len(cells))
    ax = axes[0][0]
    ax.bar(x - 0.25, [census[c]["counts"].get("P-FIRST", 0) for c in cells],
           0.25, label="P-FIRST", color="tab:blue")
    ax.bar(x, [census[c]["counts"].get("MARGIN-FIRST", 0) for c in cells],
           0.25, label="MARGIN-FIRST", color="tab:red")
    ax.bar(x + 0.25, [census[c]["counts"].get("TOGETHER", 0) for c in cells],
           0.25, label="TOGETHER", color="tab:gray")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=45, ha="right")
    ax.set_title("Bar 1: the order census (e230 conventions)")
    ax.legend()
    ax = axes[0][1]
    fr = [zomb[c]["frac"] if zomb[c]["frac"] == zomb[c]["frac"] else 0
          for c in cells]
    cm = [COMMITTED["zombie_standing"][c] for c in cells]
    ax.bar(x, fr, 0.5, label="new organism", color="tab:green")
    ax.scatter(x, cm, color="k", label="committed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=45, ha="right")
    ax.set_title("Bar 5: zombie standing fractions (e232 conventions)")
    ax.legend()
    ax = axes[1][0]
    d = bars["3_T_R2"]["detail"]
    if d:
        ax.scatter([dd["cell"] for dd in d], [dd["R2"] for dd in d],
                   color="tab:purple", zorder=3)
        ax.axhspan(0.5, 0.8, alpha=0.15, color="tab:purple", label="[0.5,0.8]")
        ax.set_ylim(0, 1)
        ax.set_title("Bar 3: T-R2 (e238 conventions)")
        ax.legend()
    ax = axes[1][1]
    s = bars["2_SEAT_Z_DIRECTION"]["detail"]
    for w, colr in (("w1", "tab:blue"), ("w2", "tab:red")):
        if w not in s:
            continue
        rr = s[w]["reads"]
        sts = sorted(rr, key=int)
        ax.plot([int(k) for k in sts], [rr[k]["A_dying"] for k in sts],
                "-o", color=colr, label=f"{w} dying ({s[w]['dying']})")
        ax.plot([int(k) for k in sts], [rr[k]["A_holder"] for k in sts],
                "--s", color=colr, label=f"{w} holder ({s[w]['holder']})")
    ax.set_xlabel("wash step")
    ax.set_ylabel("alignment A(w,s,i)")
    ax.set_title("Bar 2: the seat (e226 conventions)")
    ax.legend(fontsize=7)
    fig.suptitle(f"e248 THE ORGANISM REPLICATE — verdict: {verdict}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(RUN / "e248_adjudication.png", dpi=130)
    plt.close(fig)


def _notes_draft(verdict, n_pass, bars, gates) -> str:
    L = [
        "## e248 — THE ORGANISM REPLICATE (draft NOTES entry; the "
        "coordinator folds)",
        "",
        f"VERDICT: {verdict} ({n_pass}/5 primaries; gates "
        + ("all pass" if all(gates.values()) else
           str([k for k, v in gates.items() if not v])) + ").",
        "",
        "WHAT WE DID: the R64 dispatch — a fresh ~116M-class char-LM",
        "(116,279,296 params, 12L/14H/896/512ctx, data/input.txt, the",
        "g-series recipe scaled, val-anchored cosine with the U-turn",
        "guard), the fact-install (the e182c-family recipe on a spliced",
        "teach stream, the frozen dose ladder), the committed 54-probe",
        "battery (e182/e182c/e182c2 pools verbatim, the char-dialect",
        "answer read), TWO washes (the e182 recipe verbatim at char",
        "level, seeds 24811/24812), and the eight frozen bars re-read at",
        "matched conventions (registration commit precedes all compute;",
        "lab/e248_organism_replicate.py; runs/e248/).",
        "",
        "WHAT WE SAW (per bar):",
    ]
    for k, v in bars.items():
        L.append(f"- {k}: {'PASS' if v.get('pass') else 'FAIL'} — "
                 + json.dumps({kk: vv for kk, vv in v.items()
                               if kk != "detail"}, default=float)[:380])
    L += [
        "",
        "HONESTY: n=1 fresh organism; char-level/shakespeare vs the",
        "committed archive's BPE/web-corpus — the bars are instrument",
        "re-reads at matched conventions, not distribution-matched",
        "replications; every miss is a finding about organism-",
        "specificity. The wash-depth and install ladders (if fired) are",
        "pre-registered contingencies, disclosed per rung. Nothing",
        "guaranteed.",
    ]
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="e248 the organism replicate")
    ap.add_argument("cmd", choices=["freeze", "build", "train", "install",
                                    "wash", "read", "adjudicate"])
    args = ap.parse_args()
    {"freeze": cmd_freeze, "build": cmd_build, "train": cmd_train,
     "install": cmd_install, "wash": cmd_wash, "read": cmd_read,
     "adjudicate": cmd_adjudicate}[args.cmd]()


if __name__ == "__main__":
    main()
