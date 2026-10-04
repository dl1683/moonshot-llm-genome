"""E251 — THE LN-REDISTRIBUTION CONTROL (eval-only, CPU, minutes).

THE CONFOUND UNDER TEST (agy consult #001, part (c) — scratch/agy_consults/
agy_consult_001.md; the registered ask, now discharged): LayerNorm normalizes
the residual stream's TOTAL variance — if the wash destroys the fact's aligned
beliefs, LN may MECHANICALLY redistribute the variance budget to whatever
survives, inflating ALL margins (zero-sum bookkeeping) rather than the fact
battery's margin specifically (a constructive forge). The lab's OWN W001
carries the seed of the worry ("LN normalizes the STREAM TOTAL per position …
scale the OTHER contributions down and the per-entry floor drops"). E242/T220
committed the fact battery's median margin GROWTH through the wall's flat
phase: 0.979 -> 1.167 sigma (+19.2%; margin_decline_pct -19.154) — the
"commitment-forge" headline. THE FORGE MAY BE ZERO-SUM LN ARITHMETIC.

THE CELL (the dispatch letter; the agy consult's registered desk control on
e242's committed states): (1) load the SAME g1c W1 committed wash states e242
used ({t0, +1, +2, +4, +10, +50, +100, +200, +300}; the registered grid
{t0,+1,+2,+10,+50,+100,+300} adjudicates, {+4,+200} ride as TEXTURE co-reports
— e242's own convention), bit-exact per its gates; (2) run THE MARGIN
instrument (e228/e229/e242's margin_pass, MODULE-IMPORTED, arithmetic
untouched) on the CTRL battery — the val-carried word/sentence completions
(the g-series' own ctrl conventions: the val split, name-free per the neutral
bank's forbidden list, draw seed = G1.R_EVAL_SEED 26502 — the e065 val-bank
seed; context length 118 = the ruler's own g-12 geometry; stratified 30 word
completions + 30 sentence completions; NO 124M battery is computed or quoted
on the x-side) — at every state; (3) THE READ: the ctrl battery's median
margin trajectory vs the fact battery's COMMITTED one (runs/e242/metrics.json,
hard-bound); (4) the co-read: e043's OWN held-30 convention (host_occ[60:90]
under SPLICE_RNG 24301 — the install's held-out originals, never trained in
any form), read on the same states as a second, geometry-matched ctrl.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any compute;
the script is committed at birth; adjudicate against exactly this; no bar
shopping):
  - FORGE-REAL — "the fact battery's +19.2% median growth EXCEEDS the ctrl
    battery's growth by >= 2x (e.g. ctrl <= ~10%) — the thickening is
    fact-specific; the constructive forge survives its sharpest test; T220
    stands with the control cited"
  - ZERO-SUM-LN — "the ctrl battery's margins grew within 25% of the fact
    battery's rate (both ~15-25%) — the thickening is generic; the forge
    DOWNGRADES to LN variance redistribution (agy's confound fires); T220/
    W038's law-4 amended at the claim sites"
  - MIXED — "the trajectories verbatim, both batteries, no narrative
    inflation"

OPERATIONALIZATIONS (frozen BEFORE compute; they fix the clauses, they do
not move the bars):
  * THE FACT SIDE is e242's COMMITTED trajectory, read at runtime from
    runs/e242/metrics.json and hard-bound to the literals below (Rule 12):
    fact_growth_pct = 100 x (med_fact(+300)/med_fact(t0) - 1) = +19.1542%.
    This cell ALSO re-measures the fact battery on the same loads as a
    CERTIFICATION (G_FACT_REPRO, tol 1e-7 — same module-imported instrument,
    same code path, same device => T204 bit-exact expectation); the
    re-measure NEVER adjudicates (the dispatch's letter names the committed
    trajectory as the read's fact side).
  * THE CTRL SIDE is this cell's fresh measurement: ctrl_growth_pct =
    100 x (med_ctrl(+300)/med_ctrl(t0) - 1), med = the battery MEDIAN
    margin_sigma (e228/e229/e242's own primary statistic).
  * FORGE-REAL := (all hard gates PASS) AND fact_growth_pct >=
    2 x ctrl_growth_pct. Fires trivially when ctrl_growth <= 0 (no ctrl
    thickening at all is the extreme fact-specific case — the clause's
    '>=' reads it as written).
  * ZERO-SUM-LN := (all hard gates PASS) AND ctrl_growth_pct >=
    0.75 x fact_growth_pct ("grew within 25% of the fact battery's rate").
  * MIXED := otherwise, INCLUDING any hard-gate failure: if the loads or
    the instrument fail their certifications, the bars STAND DOWN (e242's
    stand-down precedent) -> MIXED with the trajectories verbatim and the
    failure disclosed. With gates passing and fact_growth > 0 the first two
    are mutually exclusive by construction (0.5f < 0.75f); asserted.
  * THE CTRL BATTERY (frozen construction, new for the 2.74M family — no
    committed prior exists at this scale; the 124M ctrl battery is
    dialect-forbidden by the dispatch): 60 probes drawn from the corpus VAL
    split; candidate answer positions i ~ torch.randint(280, len(val)-1)
    under Generator(26502) (= G1.R_EVAL_SEED, the e065/g-series val-bank
    seed); the 119-char window [i-118, i] must contain none of
    E225.ANCHOR_FORBIDDEN (the neutral bank's own filter); WORD stratum :=
    prev char alphabetic AND answer alphabetic (completing a word);
    SENTENCE stratum := prev char whitespace AND char-before-that in ".!?"
    AND answer alphabetic (starting a new sentence's first word); 30 + 30;
    ans_id = stoi[the true val next char]; context = val_text[i-118:i]
    (geometry-matched to the fact battery's 118-char g-12 ruler contexts).
  * THE HELD-30 BATTERY (frozen construction, e043's own convention):
    host_occ enumerated and shuffled EXACTLY as e242/e229's P0a (E43.find_occ
    + E43.SPLICE_RNG 24301); held_occ = host_occ[60:90]; context =
    train_text[p-118:p] (the same g-12 geometry); ans = the incumbent host's
    first char. CO-REPORT ONLY — never adjudicates (its answers span only
    {F, E}: a coarse ctrl, disclosed).
  * Aggregate = MEDIAN margin_sigma per battery per state; co-reports:
    p25/mean/min, frac below T204's ~0.05 sigma flip zone, frac
    argmax==true-char (the organism's continuation accuracy), the word/
    sentence strata medians, and growth sensitivity at the +50/+100
    endpoints (never adjudicated).

CHECKS (the dispatch's letter):
  * The states' provenance — bit-exact loads vs e242's gates: g1c_root.pt
    via e225.load_body gated by e225's G_ROOT form (param count 2,739,072 +
    CPU battery read vs the committed root_gm12, tol 5e-3); every washed
    state via g1's evl_load (settle+disarm) with (i) the light g-12
    re-probe gated against g1c's committed adjudication.wall.W1.g_m12
    (tol 5e-3), (ii) the resume ckpt inventory gates (step 300, wall_R 0.7
    fp32-toleranced, sds steps, sds[300] bit-identical to the final body),
    (iii) each state's body_flat_md5 asserted EXACTLY EQUAL to e242's
    committed per-state md5s, and (iv) this cell's fact-battery margin
    medians gated against e242's committed medians (G_FACT_REPRO, tol 1e-7).
  * The same margin convention: e228.margin_pass MODULE-IMPORTED via e229's
    _E228NetShim adapter (arithmetic untouched) — the exact instrument of
    e228/e229/e242.
  * The ctrl battery's dialect correctness: g-series conventions only (the
    val split, the neutral bank's forbidden filter, R_EVAL_SEED, the 118-char
    ruler geometry); NO 124M battery computed; the only external quote is
    e242's committed trajectory (same lineage, same instrument).
  * n=1 lineage, ONE wall commit (R=0.7), ONE wash draw (seed 10902 — g2e's
    held-seed convention), n=1 deterministic reads per state (T204); the
    other wall lineages are NOT read here. Nothing guaranteed.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch;
e246 owns the GPU lane — never claimed; e248 is desk — load-polite); torch
threads 4 (g1's import resets to 8 — reset after, e242's convention); load
checks recorded per burst; one state = one eval burst (3 batteries x
margin_pass + the light dial); minutes total; progressive metrics.json
writes after every phase.

Outputs: runs/e251/{metrics.json (PROGRESSIVE),
e251_ln_confound_control.png}. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e251_ln_confound_control.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e242's convention)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # e228's offline convention (imported)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                     # noqa: E402 — the p25 convention
import torch                                           # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, run_dir, save_json  # noqa: E402

import e043_install as E43                             # noqa: E402 — REPO, find_occ, SPLICE_RNG, jsonable
import e225_one_currency as E225                       # noqa: E402 — the roster + battery constants
import e228_margin_landscape as E228                   # noqa: E402 — THE margin instrument (VERBATIM)
import e229_wall_currency as E229                      # noqa: E402 — the shim + the P0a pattern
import g1b_continuity as GB                            # noqa: E402 — the 2.74M patch (BEFORE G1)
import g1_anchored_ball as G1                          # noqa: E402 — evl_load (settle+disarm), MAINTAIN_BAR

torch.set_num_threads(4)                                # the dispatch envelope (g1's import resets to 8)

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
import textwrap                                         # noqa: E402

CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e251 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------ frozen constants
RUNS = E43.REPO / "runs"
CKPT_DIR = GB.CKPT_DIR
G1C_ROOT_CK = "g1c_root.pt"
G1C_W1_RESUME_CK = "g1c_W1_resume.pt"
G1C_METRICS = RUNS / "g1c_root" / "metrics.json"
E242_METRICS = RUNS / "e242" / "metrics.json"

STATE_GRID: tuple[int, ...] = (1, 2, 10, 50, 100, 300)   # the registered grid (adjudicates)
TEXTURE_STATES: tuple[int, ...] = (4, 200)               # committed; texture co-reports only
G_READ_TOL = 5e-3           # e225/e229/e242's G_ROOT tolerance (CPU read vs committed reads)
T204_FLIP_SIGMA = 0.05      # T204's batch-shape flip threshold (co-report dial)
FACT_REPRO_TOL = 1e-7       # G_FACT_REPRO: same code path, same device -> bit-exact expected
CTRL_CTX = 118              # the ruler's own g-12 context length (E225.PRE - 12)
CTRL_N_PER = 30             # 30 word + 30 sentence = 60 probes (the fact battery's n)

# g1c's committed record, HARD-BOUND (e242's own literals; read at runtime
# from runs/g1c_root/metrics.json and asserted against these; Rule 12).
G1C_VERDICT = "ROOT-WALL-HOLDS"
G1C_ROOT_GM12 = 0.9026340246200562
G1C_W1_GM12 = {
    1: 0.821441113948822, 2: 0.9093567132949829, 4: 0.8956068754196167,
    10: 0.9277286529541016, 50: 0.9397175908088684, 100: 0.9368361234664917,
    200: 0.9264556169509888, 300: 0.9405527114868164,
}
G1C_W1_MIN = 0.821441113948822
G1C_ROOT_FLAT_MD5 = "a7f02b367c5342535aecfc814d780631"   # e242's committed root flat md5
G1C_W1_SHA16 = "650e631c2dbcf44f"                        # e242's committed resume-ckpt sha256_16
G1C_BODY_MDS = {                                         # e242's committed per-state body md5s
    1: "100c15c07c3fc34fb1d40ceee9494613", 2: "756360f07eae3d1f32581847e49f46b3",
    4: "15ee413d45785475ac3a18a178ba5d9d", 10: "464069503316930cef3e98234fbff5d9",
    50: "8cb39e128291db825da066bb07da8b7e", 100: "3ee37433c15b8e2c439b7eeb2423e10f",
    200: "29f8b5b49ab6af621a721533ecf88b74", 300: "d5d1bb6894944a9c877c9aa3964c61ba",
}

# e242's committed fact-battery margin trajectory, HARD-BOUND (the read's
# fact side); read at runtime from runs/e242/metrics.json and asserted.
E242_MD5 = "9bf5ae17b3091673f47e7c2a0845b4ab"
E242_VERDICT = "FLAT-COMMITMENT"
E242_MEDIANS = {   # state -> committed battery MEDIAN margin_sigma (t0 = step 0)
    0: 0.9794073282786698, 1: 0.9611918699600377, 2: 0.9933262426349674,
    4: 0.9632744808830553, 10: 1.0801870072706619, 50: 1.1753321683243787,
    100: 1.1397841935007293, 200: 1.0717460701295862, 300: 1.1670049513746559,
}
E242_P25S = {
    0: 0.7820456853101319, 1: 0.630357457497744, 2: 0.7813450446804553,
    4: 0.7017392081797632, 10: 0.8344954517158819, 50: 0.8454027931242046,
    100: 0.8669251422951434, 200: 0.7606515404518548, 300: 0.8661583427179607,
}
E242_DECLINE_PCT = -19.15419842995183                    # committed margin_decline_pct

REGISTERED_BARS = {
    "FORGE-REAL": 'FORGE-REAL — "the fact battery\'s +19.2% median growth '
        'EXCEEDS the ctrl battery\'s growth by >= 2x (e.g. ctrl <= ~10%) — '
        'the thickening is fact-specific; the constructive forge survives its '
        'sharpest test; T220 stands with the control cited"',
    "ZERO-SUM-LN": 'ZERO-SUM-LN — "the ctrl battery\'s margins grew within '
        '25% of the fact battery\'s rate (both ~15-25%) — the thickening is '
        'generic; the forge DOWNGRADES to LN variance redistribution (agy\'s '
        'confound fires); T220/W038\'s law-4 amended at the claim sites"',
    "MIXED": 'MIXED — "the trajectories verbatim, both batteries, no '
        'narrative inflation"',
    "registration": "bars frozen VERBATIM from the dispatch brief in the "
                    "module docstring BEFORE any compute (script committed "
                    "at birth); adjudicate against exactly this; no bar "
                    "shopping.",
    "clause_fixes":
        "fact side = e242's COMMITTED trajectory (runs/e242/metrics.json, "
        "hard-bound): fact_growth_pct = 100*(med_fact(+300)/med_fact(t0) - 1) "
        "= +19.1542%; ctrl side = this cell's fresh median-margin trajectory "
        "on the same states through the same module-imported instrument: "
        "ctrl_growth_pct = 100*(med_ctrl(+300)/med_ctrl(t0) - 1); FORGE-REAL "
        ":= (hard gates PASS) AND fact_growth_pct >= 2*ctrl_growth_pct (fires "
        "trivially at ctrl_growth <= 0); ZERO-SUM-LN := (hard gates PASS) AND "
        "ctrl_growth_pct >= 0.75*fact_growth_pct; MIXED := otherwise, "
        "INCLUDING any hard-gate failure (the bars stand down, e242's "
        "precedent); mutually exclusive under gates-PASS with fact_growth > 0 "
        "(0.5f < 0.75f); asserted.",
}

deviations: list[str] = [
    "EVAL-ONLY on g1c's committed artifacts (the pristine root + the W1 "
    "resume ckpt's embedded wash states) and e242's committed trajectory; no "
    "training, no wash run here; CPU-only by dispatch (e246 owns the GPU "
    "lane); load-polite vs e248 (threads 4, load checks, one state = one "
    "eval burst).",
    "The CTRL battery is a NEW construction for the 2.74M family — no "
    "committed ctrl battery exists at this scale, and the 124M ctrl battery "
    "(e214/e228's fact/ctrl/near/tmpl) is dialect-forbidden by the dispatch. "
    "Built strictly from g-series conventions: the corpus val split, the "
    "neutral bank's forbidden filter (E225.ANCHOR_FORBIDDEN), draw seed "
    "G1.R_EVAL_SEED 26502 (the e065 val-bank seed), context length 118 (the "
    "ruler's own g-12 geometry), stratified 30 word + 30 sentence "
    "completions; gated by G_CTRL.",
    "The fact battery is RE-MEASURED on the same loads as a certification "
    "(G_FACT_REPRO vs e242's committed medians, tol 1e-7); the adjudication's "
    "fact side remains the COMMITTED trajectory (the dispatch's letter) — "
    "the re-measure never adjudicates.",
    "The held-30 battery (e043's own convention: host_occ[60:90] under "
    "SPLICE_RNG 24301, g-12 geometry, incumbent-first-char answers) rides as "
    "a CO-REPORT only — never adjudicates (its answers span only {F, E}).",
    "Importing e228 (via e229) opens runs/e228_run.log in append mode as a "
    "module side effect — nothing is written to it by this cell (own stdout "
    "log). g1's import resets torch threads to 8 — reset to 4 after import.",
    "The state grid: the registered {t0, +1, +2, +10, +50, +100, +300} "
    "adjudicates; the two extra committed states {+4, +200} are read as "
    "TEXTURE co-reports (disclosed; they never adjudicate — e242's own "
    "convention).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).",
]


# ------------------------------------------------------------------ helpers
def cpu_load_probe() -> float | None:
    try:
        import psutil
        return round(float(psutil.cpu_percent(interval=1.0)), 1)
    except Exception:                                      # noqa: BLE001
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).LoadPercentage"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            return float(out) if out else None
        except Exception:                                  # noqa: BLE001
            return None


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                      # noqa: BLE001
        return "unavailable"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def body_flat_md5(sd: dict) -> str:
    """md5 of a state dict's BODY tensors (anch__ buffers excluded), in
    state_dict order — e242's function VERBATIM (provenance for the states)."""
    flat = torch.cat([sd[k].reshape(-1).float()
                      for k in sorted(sd) if not k.startswith("anch__")])
    return hashlib.md5(flat.numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e251")
    metrics: dict = {
        "experiment": "e251_ln_confound_control",
        "phase": "THE LN-REDISTRIBUTION CONTROL (agy consult #001 (c)) — did "
                 "the ctrl battery's margins thicken ~19% too on e242's "
                 "committed wash states, or is the forge fact-specific?",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": False,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; e246 owns "
                      "the GPU lane — never claimed; e248 desk — load-polite)",
            "torch_threads": torch.get_num_threads(),
            "bursts": "one state = one eval burst (3 batteries x margin_pass "
                      "+ the light dial)",
            "n_evals": "n=1 per state per battery (deterministic, T204)",
            "cpu_load_pct_at_launch": cpu_load_probe(),
        },
        "deviations": deviations,
        "builds_on": [
            "agy consult #001 part (c) (scratch/agy_consults/agy_consult_001.md "
            "— the LN variance-redistribution confound on e242's thickening; "
            "this cell IS its registered desk control)",
            "W001 (the lab's own LN observation: LN normalizes the stream "
            "TOTAL — the seed of the confound)",
            "T220 / e242 (the commitment-forge claim under test: +19.2% "
            "median margin growth through the wall's flat phase; the "
            "committed trajectory + the loading conventions this cell reuses)",
            "T207 / e228 + T208 / e229 (the argmax-margin instrument, "
            "module-imported verbatim; the shim; the P0a pattern)",
            "e043 (the held-30 convention — the install's own held-out "
            "originals, host_occ[60:90] under SPLICE_RNG 24301)",
            "T181 / g1c (the fresh root lineage: ROOT-WALL-HOLDS, the "
            "committed W1 wash states this cell reads)",
            "T204 / x3 (determinism + the ~0.05 sigma flip zone)",
        ],
        "whats_new": [
            "the FIRST non-fact margin battery ever read on the wall's wash "
            "states (the 2.74M family's first ctrl battery: val-carried "
            "word/sentence completions, 30+30, g-series conventions) — every "
            "prior margin read on this lineage probed the installed fact",
            "the LN zero-sum bookkeeping test of the forge claim: the ctrl "
            "battery's median-margin trajectory vs the fact battery's "
            "committed +19.2%, on bit-certified identical states through the "
            "identical module-imported instrument",
            "the held-30 co-read (e043's held-out originals) on the same "
            "states — a geometry-matched second ctrl",
        ],
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase_note"] = note
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"WROTE partial metrics ({note})")

    log(f"E251 — THE LN-REDISTRIBUTION CONTROL -> {rd}")

    # ================= P0a: the batteries (e242's P0a + ctrl + held) =========
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "val_zeph_count": val_text.count("ZEPH"),
                  "pass": bool(corpus_zeph == 0 and val_text.count("ZEPH") == 0)}
    assert G_NAMEFREE["pass"], f"corpus/val contains ZEPH: {G_NAMEFREE}"

    host_occ = []
    for host in E225.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E225.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    held_mix = {"FLORIZEL": sum(1 for _, h in held_occ if h == "FLORIZEL"),
                "ELIZABETH": sum(1 for _, h in held_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "held_mix": held_mix, "n_held": len(held_occ),
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                             and len(held_occ) == 30)}
    assert G_SPLICE["pass"], f"splice drift {G_SPLICE}"

    # THE FACT battery (certification re-measure; never adjudicates):
    # install-60 g-12 ruler, e242's construction VERBATIM.
    j = E225.RULER_J
    ruler_ids = torch.stack(
        [corpus.encode(train_text[p - E225.PRE - j: p]) for p, _ in install_occ])
    G_BATTERY = {
        "shapes": {"g-12": list(ruler_ids.shape)},
        "pass": bool(list(ruler_ids.shape) == [60, E225.PRE - 12]),
        "note": "install-60 g-12 ruler battery (SPLICE_RNG 24301; the "
                "family's own battery; e242's construction re-run from its "
                "module constants)",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    # THE CTRL battery: val-carried word/sentence completions (frozen above).
    g_ctrl = torch.Generator().manual_seed(int(G1.R_EVAL_SEED))
    n_val = len(val_ids)
    word_pos, sent_pos, tries = [], [], 0
    while ((len(word_pos) < CTRL_N_PER or len(sent_pos) < CTRL_N_PER)
           and tries < 500_000):
        tries += 1
        i = int(torch.randint(280, n_val - 1, (1,), generator=g_ctrl))
        ch, prev = val_text[i], val_text[i - 1]
        if not ch.isalpha() or ch == "Z":
            continue
        window = val_text[i - CTRL_CTX: i + 1]
        if any(f in window for f in E225.ANCHOR_FORBIDDEN):
            continue
        if prev.isalpha():
            if len(word_pos) < CTRL_N_PER:
                word_pos.append(i)
        elif prev.isspace() and i >= 2 and val_text[i - 2] in ".!?":
            if len(sent_pos) < CTRL_N_PER:
                sent_pos.append(i)
    ctrl_pos = word_pos + sent_pos
    ctrl_stratum = ["word"] * len(word_pos) + ["sentence"] * len(sent_pos)
    ctrl_probes = [{"ids": corpus.encode(val_text[i - CTRL_CTX: i]),
                    "fact": f"ctrl{k:02d}@{s}", "relation": f"ctrl_{s}",
                    "ans_id": stoi[val_text[i]], "pos": i}
                   for k, (i, s) in enumerate(zip(ctrl_pos, ctrl_stratum))]
    ctrl_ans_chars = [val_text[i] for i in ctrl_pos]
    G_CTRL = {
        "n_word": len(word_pos), "n_sentence": len(sent_pos),
        "expected_per_stratum": CTRL_N_PER, "tries": tries,
        "ctx_len": CTRL_CTX,
        "all_ctx_len_118": bool(all(p["ids"].shape[0] == CTRL_CTX
                                    for p in ctrl_probes)),
        "all_answers_alpha": bool(all(c.isalpha() and c != "Z"
                                      for c in ctrl_ans_chars)),
        "forbidden_free": bool(not any(
            f in val_text[i - CTRL_CTX: i + 1]
            for i in ctrl_pos for f in E225.ANCHOR_FORBIDDEN)),
        "positions_unique": bool(len(set(ctrl_pos)) == len(ctrl_pos)),
        "answer_char_hist": {c: ctrl_ans_chars.count(c)
                             for c in sorted(set(ctrl_ans_chars))},
        "source": "the corpus VAL split (last 10% of data/input.txt)",
        "draw_seed": int(G1.R_EVAL_SEED),
        "pass": bool(len(word_pos) == CTRL_N_PER
                     and len(sent_pos) == CTRL_N_PER
                     and all(p["ids"].shape[0] == CTRL_CTX for p in ctrl_probes)
                     and all(c.isalpha() and c != "Z" for c in ctrl_ans_chars)
                     and len(set(ctrl_pos)) == len(ctrl_pos)),
        "note": "the 2.74M family's first ctrl battery: val-carried word/"
                "sentence completions, 30+30, name-free per the neutral "
                "bank's filter, draw seed = G1.R_EVAL_SEED (the e065 "
                "val-bank seed), context 118 = the ruler's g-12 geometry; "
                "NO 124M battery computed here",
    }
    assert G_CTRL["pass"], f"ctrl battery construction FAILED: {G_CTRL}"

    # THE HELD-30 battery (e043's convention; co-report only).
    held_probes = [{"ids": corpus.encode(train_text[p - E225.PRE - j: p]),
                    "fact": f"held{k:02d}@{h[:4]}", "relation": "held30_incumbent",
                    "ans_id": stoi[h[0]], "pos": p, "host": h}
                   for k, (p, h) in enumerate(held_occ)]
    G_HELD = {
        "n": len(held_probes), "expected": 30,
        "mix": held_mix,
        "all_ctx_len_118": bool(all(p["ids"].shape[0] == CTRL_CTX
                                    for p in held_probes)),
        "answers": sorted({p["host"][0] for p in held_probes}),
        "note": "e043's own held convention (host_occ[60:90] under "
                "SPLICE_RNG 24301 — the install's held-out originals, never "
                "trained in any form); g-12 geometry; answer = the "
                "incumbent host's first char; CO-REPORT ONLY (a coarse ctrl: "
                "its answers span only {F, E})",
        "pass": bool(len(held_probes) == 30
                     and all(p["ids"].shape[0] == CTRL_CTX
                             for p in held_probes)),
    }
    assert G_HELD["pass"], f"held battery construction FAILED: {G_HELD}"

    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_CTRL": G_CTRL,
                        "G_HELD": G_HELD}
    metrics["batteries"] = {
        "fact_g-12": {"n": 60, "ans": "Z", "role": "certification re-measure "
                      "vs e242's committed trajectory (never adjudicates)"},
        "ctrl": {"n": len(ctrl_probes), "strata": {"word": len(word_pos),
                                                   "sentence": len(sent_pos)},
                 "probes": [{"fact": p["fact"], "pos": p["pos"],
                             "ans_char": val_text[p["pos"]]}
                            for p in ctrl_probes],
                 "role": "THE REGISTERED READ (the dispatch's ctrl battery)"},
        "held30": {"n": len(held_probes),
                   "probes": [{"fact": p["fact"], "pos": p["pos"],
                               "host": p["host"]} for p in held_probes],
                   "role": "co-read (e043's held convention; never "
                           "adjudicates)"},
    }
    log(f"P0a: batteries built — fact 60 (splice 19+41); ctrl "
        f"{len(word_pos)}W+{len(sent_pos)}S (tries {tries}); held "
        f"{len(held_probes)} (mix {held_mix})")
    write_partial("P0a battery gates PASSED")

    # ================= P0b: G_PARENTS — committed records hard-bound ========
    for pth in (G1C_METRICS, E242_METRICS):
        if not pth.exists():
            raise RuntimeError(f"missing parent record: {pth}")
    g1c = json.loads(G1C_METRICS.read_text(encoding="utf-8"))
    e242m = json.loads(E242_METRICS.read_text(encoding="utf-8"))

    g1c_w1 = {int(k): v for k, v in
              g1c["adjudication"]["wall"]["W1"]["g_m12"].items()}
    g1c_root = g1c["root_build"]["root_cells"]["gm12"]
    ok_g1c = (g1c["adjudication"]["verdict"] == G1C_VERDICT
              and abs(g1c_root - G1C_ROOT_GM12) < 1e-12
              and all(abs(g1c_w1[s] - G1C_W1_GM12[s]) < 1e-12 for s in G1C_W1_GM12)
              and abs(g1c["adjudication"]["wall"]["W1"]["min_gm12"]
                      - G1C_W1_MIN) < 1e-12)

    e242_md5 = md5of(E242_METRICS)
    traj242 = {r["step"]: r for r in e242m["trajectory"]}
    ok_e242 = (e242_md5 == E242_MD5
               and e242m["adjudication"]["verdict"] == E242_VERDICT
               and set(traj242) == set(E242_MEDIANS)
               and all(abs(traj242[s]["margin_median_sigma"] - E242_MEDIANS[s])
                       < 1e-12 for s in E242_MEDIANS)
               and all(abs(traj242[s]["margin_p25_sigma"] - E242_P25S[s]) < 1e-12
                       for s in E242_P25S)
               and abs(e242m["adjudication"]["reads"]["margin_decline_pct"]
                       - E242_DECLINE_PCT) < 1e-12)

    G_PARENTS = {
        "g1c_metrics": {"path": str(G1C_METRICS), "md5": md5of(G1C_METRICS),
                        "sha256_16": sha256_of(G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"],
                        "root_gm12": g1c_root,
                        "W1_g_m12": {str(k): v for k, v in sorted(g1c_w1.items())}},
        "e242_metrics": {"path": str(E242_METRICS), "md5": e242_md5,
                         "sha256_16": sha256_of(E242_METRICS),
                         "verdict": e242m["adjudication"]["verdict"],
                         "committed_fact_medians":
                             {str(s): traj242[s]["margin_median_sigma"]
                              for s in sorted(traj242)},
                         "committed_fact_p25s":
                             {str(s): traj242[s]["margin_p25_sigma"]
                              for s in sorted(traj242)},
                         "committed_margin_decline_pct":
                             e242m["adjudication"]["reads"]["margin_decline_pct"]},
        "pass": bool(ok_g1c and ok_e242),
        "note": "g1c's committed record (the state certification targets) and "
                "e242's committed fact-battery trajectory (THE READ's fact "
                "side) read at runtime and asserted against the literals "
                "frozen in this script pre-compute (Rule 12)",
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — g1c {G1C_VERDICT}; e242 {E242_VERDICT} "
        f"(md5 {e242_md5}; fact median t0 "
        f"{E242_MEDIANS[0]:.4f}s -> +300 {E242_MEDIANS[300]:.4f}s)")
    write_partial("P0b parent gates PASSED")

    # ================= P1: the states — load, certify, measure ==============
    all_steps = tuple(sorted(STATE_GRID + TEXTURE_STATES))   # (1,2,4,10,50,100,200,300)
    wck = torch.load(CKPT_DIR / G1C_W1_RESUME_CK, map_location=CPU,
                     weights_only=False)
    sds_keys = sorted(int(k) for k in wck["sds"].keys())
    body_keys_final = {k: v for k, v in wck["model"].items()
                       if not str(k).startswith("anch__")}
    sds300 = {k: v for k, v in wck["sds"][300].items()
              if not str(k).startswith("anch__")}
    max_diff_300 = max(float(torch.max(torch.abs(sds300[k] - body_keys_final[k])))
                       for k in body_keys_final)
    state_body_mds = {s: body_flat_md5(wck["sds"][s]) for s in all_steps}
    G_STATES_A = {
        "checkpoint": f"runs/checkpoints/{G1C_W1_RESUME_CK}",
        "sha256_16": sha256_of(CKPT_DIR / G1C_W1_RESUME_CK),
        "sha256_16_matches_e242": bool(
            sha256_of(CKPT_DIR / G1C_W1_RESUME_CK) == G1C_W1_SHA16),
        "step_field": int(wck["step"]),
        "wall_R": float(wck["wall_R"]),
        "sds_steps": sds_keys,
        "sds300_bit_identical_to_final_model": bool(max_diff_300 == 0.0),
        "expected_sds_steps": list(all_steps),
        "body_flat_md5s_match_e242_committed": bool(all(
            state_body_mds[s] == G1C_BODY_MDS[s] for s in all_steps)),
        "pass": bool(int(wck["step"]) == 300
                     and abs(float(wck["wall_R"]) - 0.7) < 1e-6  # fp32 round-trip of 0.7
                     and tuple(sds_keys) == all_steps and max_diff_300 == 0.0
                     and all(state_body_mds[s] == G1C_BODY_MDS[s]
                             for s in all_steps)
                     and sha256_of(CKPT_DIR / G1C_W1_RESUME_CK) == G1C_W1_SHA16),
    }
    assert G_STATES_A["pass"], f"W1 resume inventory gate FAILED: {G_STATES_A}"

    # the t0 root — e225's G_ROOT form (+ e242's committed root flat md5)
    root_net, root_sd, root_meta = E225.load_body(CKPT_DIR / G1C_ROOT_CK, Cfg())
    n_par = sum(p.numel() for p in root_net.parameters())
    theta0 = E225.flat_params(root_net)
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()
    root_read = E225.battery_cell(root_net, ruler_ids, zid)["mean_pz"]
    rdev = abs(root_read - G1C_ROOT_GM12)
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{G1C_ROOT_CK}",
        "n_params": n_par, "expected_params": 2_739_072,
        "battery_read_measured": root_read,
        "battery_read_committed": G1C_ROOT_GM12,
        "abs_diff": rdev, "tol": G_READ_TOL,
        "flat_md5": root_md5,
        "flat_md5_matches_e242": bool(root_md5 == G1C_ROOT_FLAT_MD5),
        "pass": bool(n_par == 2_739_072 and rdev < G_READ_TOL
                     and root_md5 == G1C_ROOT_FLAT_MD5),
    }
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    log(f"P1: G_ROOT[{G1C_ROOT_CK}] PASS — {n_par} params; battery "
        f"{root_read:.10f} vs committed {G1C_ROOT_GM12:.10f} "
        f"(|d| {rdev:.1e}); flat md5 == e242's")
    metrics["gates"]["G_ROOT"] = G_ROOT
    metrics["gates"]["G_STATES_A"] = G_STATES_A

    def fact_battery() -> list[dict]:
        return [{"ids": ruler_ids[i: i + 1],
                 "fact": f"install{i:02d}@g-12",
                 "relation": "install60_g-12",
                 "ans_id": zid}
                for i in range(ruler_ids.shape[0])]

    fbatt = fact_battery()

    def margin_aggregates(rec: dict) -> dict:
        ms = [r["margin_sigma"] for r in rec["probes"]]
        return {
            "n": len(ms),
            "median": rec["median_margin_sigma"],
            "mean": rec["mean_margin_sigma"],
            "p25": float(np.percentile(ms, 25)),
            "min": rec["min_margin_sigma"],
            "frac_below_T204_flip_zone":
                float(sum(1 for m in ms if m < T204_FLIP_SIGMA) / len(ms)),
            "frac_argmax_answer": rec["frac_argmax_answer"],
            "mean_p": rec["mean_p"],
        }

    def measure_state(tag: str, net) -> dict:
        """One state = one eval burst: the three batteries through e228's
        margin_pass (module-imported, via e229's shim) + the light dial."""
        loadpct = cpu_load_probe()
        shim = E229._E228NetShim(net)
        out = {"tag": tag, "cpu_load_pct": loadpct,
               "gm12_ruler": G1.battery_cell(net, ruler_ids, zid)["mean_pz"]}
        for name, batt in (("fact", fbatt), ("ctrl", ctrl_probes),
                           ("held30", held_probes)):
            rec = E228.margin_pass(shim, batt)
            out[name] = {"aggregate": margin_aggregates(rec),
                         "probes": [{"fact": r["fact"], "p": r["p"],
                                     "margin_sigma": r["margin_sigma"],
                                     "argmax_is_answer": r["argmax_is_answer"],
                                     "top1_id": r["top1_id"],
                                     "top2_id": r["top2_id"]}
                                    for r in rec["probes"]]}
        # the ctrl strata (co-reports)
        for stratum in ("word", "sentence"):
            ms = [p["margin_sigma"] for p in out["ctrl"]["probes"]
                  if p["fact"].endswith(f"@{stratum}")]
            out["ctrl"][f"stratum_{stratum}"] = {
                "n": len(ms),
                "median": float(np.median(ms)),
                "mean": float(np.mean(ms)),
            }
        return out

    cells: dict = {}

    # t0
    cells["t0"] = measure_state("t0", root_net)
    cells["t0"]["G"] = {"kind": "pristine root (e225 load_body)",
                        "flat_md5": G_ROOT["flat_md5"]}
    log(f"  t0: ruler {cells['t0']['gm12_ruler']:.4f} | fact med "
        f"{cells['t0']['fact']['aggregate']['median']:.4f}s | ctrl med "
        f"{cells['t0']['ctrl']['aggregate']['median']:.4f}s (argmax-ans "
        f"{cells['t0']['ctrl']['aggregate']['frac_argmax_answer']:.2f}) | "
        f"held med {cells['t0']['held30']['aggregate']['median']:.4f}s")
    write_partial("P1 t0 measured (all three batteries)")
    del root_net

    # the washed states
    g_state_rows, fact_repro_rows = {}, {}
    for s in all_steps:
        net = G1.evl_load(wck["sds"][s])
        key = f"w1+{s}"
        cells[key] = measure_state(key, net)
        cells[key]["G"] = {"kind": "W1 wash state (g1 evl_load settle+disarm)",
                           "body_flat_md5": state_body_mds[s]}
        committed = G1C_W1_GM12[s]
        d = abs(cells[key]["gm12_ruler"] - committed)
        g_state_rows[s] = {"measured": cells[key]["gm12_ruler"],
                           "committed": committed, "abs_diff": d}
        fd = abs(cells[key]["fact"]["aggregate"]["median"] - E242_MEDIANS[s])
        fact_repro_rows[s] = {"this_cell": cells[key]["fact"]["aggregate"]["median"],
                              "e242_committed": E242_MEDIANS[s], "abs_diff": fd}
        role = "ADJUDICATES" if s in STATE_GRID else "texture"
        log(f"  +{s:>3} ({role}): ruler {cells[key]['gm12_ruler']:.6f} vs "
            f"committed {committed:.6f} (|d| {d:.1e}) | fact med "
            f"{cells[key]['fact']['aggregate']['median']:.4f}s (repro |d| "
            f"{fd:.1e}) | ctrl med "
            f"{cells[key]['ctrl']['aggregate']['median']:.4f}s | held med "
            f"{cells[key]['held30']['aggregate']['median']:.4f}s")
        del net
        write_partial(f"P1 w1+{s} measured (all three batteries)")

    G_STATES = {
        **{k: v for k, v in G_STATES_A.items()},
        "per_state_ruler_reads": {str(k): v for k, v in sorted(g_state_rows.items())},
        "tol": G_READ_TOL,
        "max_abs_diff": max(v["abs_diff"] for v in g_state_rows.values()),
        "pass": bool(G_STATES_A["pass"] and all(
            v["abs_diff"] < G_READ_TOL for v in g_state_rows.values())),
        "note": "every W1 state's light g-12 re-probe (this cell's forwards) "
                "vs g1c's committed adjudication.wall.W1.g_m12, PLUS the "
                "per-state body-flat md5s and the resume ckpt's sha256 "
                "asserted EXACTLY EQUAL to e242's committed provenance — the "
                "bit-exactness certification of the loaded states",
    }
    assert G_STATES["pass"], f"state certification FAILED: {G_STATES}"
    metrics["gates"]["G_STATES"] = G_STATES

    G_FACT_REPRO = {
        "per_state": {str(s): v for s, v in sorted(fact_repro_rows.items())},
        "tol": FACT_REPRO_TOL,
        "max_abs_diff": max(v["abs_diff"] for v in fact_repro_rows.values()),
        "pass": bool(all(v["abs_diff"] < FACT_REPRO_TOL
                         for v in fact_repro_rows.values())),
        "note": "this cell's fact-battery margin medians vs e242's committed "
                "medians on the same loads, same module-imported instrument, "
                "same device — the instrument-path certification; the "
                "re-measure NEVER adjudicates (the fact side is the committed "
                "trajectory by the dispatch's letter)",
    }
    assert G_FACT_REPRO["pass"], f"fact repro FAILED: {G_FACT_REPRO}"
    metrics["gates"]["G_FACT_REPRO"] = G_FACT_REPRO
    log(f"P1 COMPLETE: all {len(all_steps)} W1 states + t0 certified (ruler "
        f"max |d| {G_STATES['max_abs_diff']:.1e}; fact-repro max |d| "
        f"{G_FACT_REPRO['max_abs_diff']:.1e}) and measured on all 3 batteries")
    write_partial("P1 COMPLETE (all states gated + 3 batteries measured)")

    # ================= P2: the trajectories + the frozen adjudication =======
    def traj_of(batt: str) -> list[dict]:
        rows = []
        for key in ["t0"] + [f"w1+{s}" for s in all_steps]:
            c = cells[key]
            step = 0 if key == "t0" else int(key.split("+")[1])
            a = c[batt]["aggregate"]
            row = {"state": key, "step": step,
                   "role": "t0" if step == 0 else
                           ("ADJUDICATES" if step in STATE_GRID else "texture"),
                   "median_sigma": a["median"], "p25_sigma": a["p25"],
                   "mean_sigma": a["mean"], "min_sigma": a["min"],
                   "frac_below_flip_zone": a["frac_below_T204_flip_zone"],
                   "frac_argmax_answer": a["frac_argmax_answer"],
                   "mean_p": a["mean_p"]}
            if batt == "ctrl":
                for stratum in ("word", "sentence"):
                    row[f"stratum_{stratum}_median"] = \
                        c["ctrl"][f"stratum_{stratum}"]["median"]
            rows.append(row)
        return rows

    metrics["trajectory"] = {
        "fact_committed_e242": [
            {"state": "t0" if s == 0 else f"w1+{s}", "step": s,
             "role": "t0" if s == 0 else
                     ("ADJUDICATES" if s in STATE_GRID else "texture"),
             "median_sigma": E242_MEDIANS[s], "p25_sigma": E242_P25S[s]}
            for s in sorted(E242_MEDIANS)],
        "fact_remeasured": traj_of("fact"),
        "ctrl": traj_of("ctrl"),
        "held30_coreport": traj_of("held30"),
    }

    # the frozen growth objects (registered grid: t0 -> +300)
    fact_growth_pct = 100.0 * (E242_MEDIANS[300] / E242_MEDIANS[0] - 1.0)
    ctrl_med0 = cells["t0"]["ctrl"]["aggregate"]["median"]
    ctrl_med300 = cells["w1+300"]["ctrl"]["aggregate"]["median"]
    ctrl_growth_pct = 100.0 * (ctrl_med300 / ctrl_med0 - 1.0)
    held_med0 = cells["t0"]["held30"]["aggregate"]["median"]
    held_med300 = cells["w1+300"]["held30"]["aggregate"]["median"]
    held_growth_pct = 100.0 * (held_med300 / held_med0 - 1.0)
    # sensitivity co-reports (never adjudicated): growth to +50/+100/+300
    def _med_at(batt: str, step: int) -> float:
        if batt == "fact_committed":
            return E242_MEDIANS[step]
        key = "t0" if step == 0 else f"w1+{step}"
        return cells[key][batt]["aggregate"]["median"]

    sens = {}
    for batt in ("fact_committed", "ctrl", "held30"):
        m0 = _med_at(batt, 0)
        for end in (50, 100, 300):
            sens[f"{batt}_growth_to_+{end}"] = \
                100.0 * (_med_at(batt, end) / m0 - 1.0)

    hard_gates = {g: bool(v.get("pass")) for g, v in metrics["gates"].items()}
    gates_ok = all(hard_gates.values())

    forge_fires = bool(gates_ok
                       and fact_growth_pct >= 2.0 * ctrl_growth_pct)
    zerosum_fires = bool(gates_ok
                         and ctrl_growth_pct >= 0.75 * fact_growth_pct)
    mixed_fires = not (forge_fires or zerosum_fires)
    if gates_ok:
        assert not (forge_fires and zerosum_fires), "bars must be exclusive"

    def _g_txt(pct: float) -> str:
        return (f"{pct:+.2f}%" + (" (GROWTH)" if pct > 0
                                  else " (no growth / shrink)"))

    if forge_fires:
        verdict = "FORGE-REAL"
        clause = (f"the fact battery's +{fact_growth_pct:.1f}% median growth "
                  f"EXCEEDS the ctrl battery's growth by >= 2x (fact "
                  f"{_g_txt(fact_growth_pct)} vs ctrl "
                  f"{_g_txt(ctrl_growth_pct)}; the 2x line sits at "
                  f"{2.0 * ctrl_growth_pct:+.2f}%) — the thickening is "
                  f"fact-specific; the constructive forge survives its "
                  f"sharpest test; T220 stands with the control cited "
                  f"(held-30 co-read {held_growth_pct:+.2f}%)")
    elif zerosum_fires:
        verdict = "ZERO-SUM-LN"
        clause = (f"the ctrl battery's margins grew within 25% of the fact "
                  f"battery's rate (ctrl {_g_txt(ctrl_growth_pct)} vs fact "
                  f"{_g_txt(fact_growth_pct)}; the 0.75x line sits at "
                  f"{0.75 * fact_growth_pct:+.2f}%) — the thickening is "
                  f"generic; the forge DOWNGRADES to LN variance "
                  f"redistribution (agy's confound fires); T220/W038's law-4 "
                  f"amended at the claim sites (held-30 co-read "
                  f"{held_growth_pct:+.2f}%)")
    else:
        why = []
        if not gates_ok:
            failed = [g for g, v in hard_gates.items() if not v]
            why.append(f"the bars STAND DOWN: hard gate(s) FAILED {failed} "
                       f"(the loads or the instrument failed their "
                       f"certifications)")
        else:
            why.append(f"the ctrl growth {ctrl_growth_pct:+.2f}% lands in the "
                       f"gap between the frozen lines (2x line "
                       f"{2.0 * ctrl_growth_pct:+.2f}% = ctrl <= 0.5x fact; "
                       f"0.75x line {0.75 * fact_growth_pct:+.2f}%) — neither "
                       f"the 2x separation nor the within-25% match")
        clause = ("; ".join(why)
                  + " — the trajectories verbatim, both batteries, no "
                  "narrative inflation")

    metrics["adjudication"] = {
        "bars": {"FORGE-REAL": {"fires": forge_fires},
                 "ZERO-SUM-LN": {"fires": zerosum_fires},
                 "MIXED": {"fires": mixed_fires}},
        "clause_fixes_applied": REGISTERED_BARS["clause_fixes"],
        "verdict": verdict, "clause": clause,
        "reads": {
            "fact_growth_pct_committed": fact_growth_pct,
            "fact_median_t0_committed": E242_MEDIANS[0],
            "fact_median_300_committed": E242_MEDIANS[300],
            "ctrl_growth_pct": ctrl_growth_pct,
            "ctrl_median_t0": ctrl_med0,
            "ctrl_median_300": ctrl_med300,
            "held30_growth_pct_coreport": held_growth_pct,
            "forge_2x_line_pct_of_fact": 50.0,
            "zerosum_075x_line_pct_of_fact": 75.0,
            "hard_gates_all_pass": gates_ok,
            "growth_sensitivity_coreports": sens,
        },
        "gates_summary": hard_gates,
    }
    log("=" * 78)
    log(f"E251 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  fact (committed) {_g_txt(fact_growth_pct)} | ctrl "
        f"{_g_txt(ctrl_growth_pct)} | held-30 {_g_txt(held_growth_pct)}")
    log("=" * 78)
    write_partial("P2 the frozen bars adjudicated")

    # ================= P3: the figure ========================================
    keys = ["t0"] + [f"w1+{s}" for s in all_steps]
    steps_lab = ["t0"] + [f"+{s}" for s in all_steps]
    xs = np.arange(len(keys))

    fact_c = [E242_MEDIANS[0]] + [E242_MEDIANS[s] for s in all_steps]
    fact_r = [cells[k]["fact"]["aggregate"]["median"] for k in keys]
    ctrl_y = [cells[k]["ctrl"]["aggregate"]["median"] for k in keys]
    ctrl_w = [cells[k]["ctrl"]["stratum_word"]["median"] for k in keys]
    ctrl_s = [cells[k]["ctrl"]["stratum_sentence"]["median"] for k in keys]
    held_y = [cells[k]["held30"]["aggregate"]["median"] for k in keys]
    adj_idx = [0] + [i for i, s in enumerate(all_steps) if s in STATE_GRID]
    tex_idx = [i for i, s in enumerate(all_steps) if s not in STATE_GRID]

    fig, axes = plt.subplots(1, 3, figsize=(21.0, 7.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})

    # (a) the two batteries side by side (same sigma units, one axis)
    ax = axes[0]
    ax.plot(xs, fact_c, "o-", color="tab:red", lw=2.0, ms=8, zorder=5,
            label="FACT battery — e242 COMMITTED median")
    ax.plot(xs, fact_r, "d", color="darkred", ms=5, zorder=6, alpha=0.9,
            label="fact re-measured (certification; should overlay)")
    ax.plot(xs, ctrl_y, "s-", color="tab:blue", lw=2.0, ms=8, zorder=5,
            label="CTRL battery — val word/sentence completions")
    ax.plot(xs, held_y, "^-", color="tab:green", lw=1.6, ms=7, zorder=4,
            alpha=0.9, label="held-30 co-read (e043's held originals)")
    for i in tex_idx:
        ax.axvline(i, color="gray", lw=6, alpha=0.08, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(steps_lab, fontsize=8)
    ax.set_xlabel("wash step (W1, wall R=0.7; shaded = texture states)")
    ax.set_ylabel("battery MEDIAN argmax margin (sigma)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.4, loc="best")
    ax.set_title(f"(a) THE TWO BATTERIES — fact {fact_growth_pct:+.1f}% vs "
                 f"ctrl {ctrl_growth_pct:+.1f}% (held {held_growth_pct:+.1f}%)",
                 fontsize=9.5)

    # (b) normalized to t0, with the frozen bar geometry
    ax = axes[1]
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    y_2x = 1.0 + 0.5 * fact_growth_pct / 100.0
    y_075 = 1.0 + 0.75 * fact_growth_pct / 100.0
    y_f = 1.0 + fact_growth_pct / 100.0
    ax.axhspan(y_2x, y_075, color="gold", alpha=0.18)
    ax.axhline(y_2x, color="tab:blue", ls="--", lw=1.2)
    ax.axhline(y_075, color="tab:purple", ls="--", lw=1.2)
    ax.axhline(y_f, color="tab:red", ls=":", lw=1.4)
    ax.annotate(f"FORGE-REAL boundary: ctrl <= 0.5x fact "
                f"({y_2x:.3f})", (0.02, y_2x),
                xycoords=("axes fraction", "data"), fontsize=7,
                color="tab:blue", va="bottom")
    ax.annotate(f"ZERO-SUM-LN boundary: ctrl >= 0.75x fact "
                f"({y_075:.3f})", (0.02, y_075),
                xycoords=("axes fraction", "data"), fontsize=7,
                color="tab:purple", va="bottom")
    ax.annotate(f"fact +{fact_growth_pct:.1f}% (committed)", (0.02, y_f),
                xycoords=("axes fraction", "data"), fontsize=7,
                color="tab:red", va="bottom")
    ax.plot(xs, [y / fact_c[0] for y in fact_c], "o-", color="tab:red",
            ms=7, lw=1.8, label="fact / fact(t0) — committed")
    ax.plot(xs, [y / ctrl_y[0] for y in ctrl_y], "s-", color="tab:blue",
            ms=7, lw=1.8, label="ctrl / ctrl(t0)")
    ax.plot(xs, [y / ctrl_w[0] for y in ctrl_w], "s:", color="tab:cyan",
            ms=4, lw=1.0, alpha=0.9, label="ctrl word stratum")
    ax.plot(xs, [y / ctrl_s[0] for y in ctrl_s], "s:", color="navy",
            ms=4, lw=1.0, alpha=0.9, label="ctrl sentence stratum")
    ax.plot(xs, [y / held_y[0] for y in held_y], "^-", color="tab:green",
            ms=6, lw=1.5, alpha=0.9, label="held-30 / held(t0)")
    ax.set_xticks(xs)
    ax.set_xticklabels(steps_lab, fontsize=8)
    ax.set_xlabel("wash step (gold band = the MIXED gap between the bars)")
    ax.set_ylabel("normalized to t0")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.2, loc="best")
    ax.set_title("(b) THE FROZEN BAR GEOMETRY — both batteries on one ruler",
                 fontsize=9.5)

    # (c) the verdict + the table
    ax = axes[2]
    ax.axis("off")
    y = 0.97
    ax.text(0.03, y, f"E251 VERDICT: {verdict}", fontsize=11.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=64, break_long_words=False)[:10]:
        ax.text(0.03, y, wd, fontsize=7.0, va="top", family="monospace")
        y -= 0.023
    y -= 0.015
    ax.text(0.03, y, "state   fact(c)  fact(r)  ctrl   ctrlW  ctrlS  held",
            fontsize=7.4, va="top", family="monospace", weight="bold")
    y -= 0.024
    for i, k in enumerate(keys):
        ax.text(0.03, y,
                f"{steps_lab[i]:>5} {fact_c[i]:8.4f} {fact_r[i]:8.4f} "
                f"{ctrl_y[i]:6.4f} {ctrl_w[i]:6.4f} {ctrl_s[i]:6.4f} "
                f"{held_y[i]:6.4f}",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.022
    y -= 0.015
    ax.text(0.03, y, f"growth t0->+300: fact {fact_growth_pct:+.2f}% "
            f"(committed) | ctrl {ctrl_growth_pct:+.2f}% | held "
            f"{held_growth_pct:+.2f}%", fontsize=7.4, va="top",
            family="monospace", color="dimgray")
    y -= 0.03
    ax.text(0.03, y, "GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}" for g, v in hard_gates.items()),
        fontsize=6.8, va="top", family="monospace")
    y -= 0.028
    ax.text(0.03, y, "the confound (agy #001c / W001): LN normalizes the "
            "stream total — if the", fontsize=7.0, va="top",
            family="monospace", color="dimgray")
    y -= 0.02
    ax.text(0.03, y, "wash kills aligned beliefs, LN may inflate ALL margins "
            "(zero-sum).", fontsize=7.0, va="top", family="monospace",
            color="dimgray")
    y -= 0.02
    ax.text(0.03, y, "The ctrl battery prices it: same states, same "
            "instrument.", fontsize=7.0, va="top", family="monospace",
            color="dimgray")

    fig.suptitle("E251 — THE LN-REDISTRIBUTION CONTROL (agy consult #001c): "
                 "the ctrl battery's margin trajectory vs the fact battery's "
                 f"committed +19.2% on g1c W1's wash states -> {verdict}",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = rd / "e251_ln_confound_control.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    # ================= P4: honesty + provenance + close ======================
    metrics["honesty"] = {
        "confound_scope": "a FORGE-REAL verdict excludes the LN "
                          "redistribution confound as the WHOLE of the "
                          "+19.2% (the ctrl battery — the same states, the "
                          "same sigma-normalized instrument — did not "
                          "thicken comparably); it does NOT price partial LN "
                          "contributions inside the fact-specific growth, "
                          "and it says nothing about the margin ABSOLUTE "
                          "levels, only the trajectories",
        "sigma_denominator_note": "margin_sigma divides by std(vocab "
                                  "logits): a generic LN-driven shrinkage of "
                                  "the denominator would inflate BOTH "
                                  "batteries identically — the instrument is "
                                  "shared, so the confound cannot hide in "
                                  "the normalization; it is exactly what "
                                  "this cell prices",
        "ctrl_battery_status": "a NEW battery for the 2.74M family (no "
                               "committed prior at this scale; the 124M ctrl "
                               "battery is dialect-forbidden); its gates are "
                               "construction gates (G_CTRL), not "
                               "reproductions — the battery is registered "
                               "here for future cells to reuse",
        "fact_side_is_committed": "the adjudication's fact side is e242's "
                                  "COMMITTED trajectory (the dispatch's "
                                  "letter), hard-bound at md5 "
                                  f"{E242_MD5}; this cell's re-measure "
                                  "(max |d| "
                                  f"{G_FACT_REPRO['max_abs_diff']:.1e}) "
                                  "certifies the path and never adjudicates",
        "n_and_scope": "ONE lineage (g1c's fresh root, ONE wall commit R=0.7, "
                       "ONE wash draw seed 10902 — g2e's held-seed "
                       "convention), ONE ctrl battery draw (seed 26502), n=1 "
                       "deterministic reads per state per battery (T204); "
                       "the heights lottery (W028/R64) stands: the thickening "
                       "could be a draw — this control tests its MECHANISM "
                       "class (fact-specific vs generic), not its "
                       "replicability across lineages",
        "determinism": "margins deterministic in-session (T204: same code "
                       "path, same device -> bit-exact); the eval batch "
                       "shape (1 x 118, single-probe forwards) is disclosed "
                       "— batch-shape re-rounding sits at the ~3e-7 texture "
                       "floor, far under every aggregate here",
        "nothing_guaranteed": "the trajectories could have landed anywhere; "
                              "the observed outcome is recorded verbatim "
                              "against the frozen bars; no bar shopping",
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "parents": {"g1c_metrics": G_PARENTS["g1c_metrics"],
                    "e242_metrics": G_PARENTS["e242_metrics"]},
        "checkpoints": {
            "t0_root": {"file": f"runs/checkpoints/{G1C_ROOT_CK}",
                        "flat_md5": G_ROOT["flat_md5"],
                        "n_params": G_ROOT["n_params"],
                        "battery_read": G_ROOT["battery_read_measured"]},
            "w1_states": {"file": f"runs/checkpoints/{G1C_W1_RESUME_CK}",
                          "sha256_16": G_STATES_A["sha256_16"],
                          "step": G_STATES_A["step_field"],
                          "wall_R": G_STATES_A["wall_R"],
                          "sds_steps": G_STATES_A["sds_steps"],
                          "sds300_bit_identical_to_final_model":
                              G_STATES_A["sds300_bit_identical_to_final_model"],
                          "per_state_body_flat_md5":
                              {f"w1+{s}": state_body_mds[s]
                               for s in all_steps}},
        },
        "machinery": {
            "margin_instrument": "lab/e228_margin_landscape.py margin_pass "
                                 "MODULE-IMPORTED via e229's _E228NetShim "
                                 "(adapter only; arithmetic untouched): "
                                 "margin_sigma = (top1-top2 logit)/"
                                 "std(vocab logits) at the answer position, "
                                 "torch std unbiased — the exact instrument "
                                 "of e228/e229/e242",
            "state_loads": "t0 via e225.load_body (the roster's own "
                           "convention); W1 states via g1's evl_load "
                           "(settle onto the ball + disarm — the lineage's "
                           "registered instrument adaptation), certified "
                           "per-state against g1c's committed W1 light-dial "
                           "record (tol 5e-3), against e242's committed "
                           "body-flat md5s (EXACT), and against e242's "
                           "committed fact-battery medians (tol 1e-7)",
            "batteries": "fact = install-60 g-12 ruler (SPLICE_RNG 24301; "
                         "certification re-measure); ctrl = val-carried "
                         "word/sentence completions 30+30 (draw seed "
                         "G1.R_EVAL_SEED 26502, E225.ANCHOR_FORBIDDEN "
                         "filter, 118-char contexts); held-30 = e043's "
                         "host_occ[60:90] held originals (g-12 geometry, "
                         "incumbent first-char answers); NO 124M battery "
                         "computed",
        },
        "eval": {"device": "cpu fp32", "threads": torch.get_num_threads(),
                 "batch_shape": "1 x 118 per probe (e228's margin_pass shape)",
                 "n_forwards": f"9 states x (60 fact + 60 ctrl + 30 held) "
                               f"margin probes + 9 light-dial batch reads = "
                               f"{9 * 150 + 9}"},
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(rd / "metrics.json"), str(png)]
    write_partial("P4 DONE (honesty + provenance + figure)")
    log(f"outputs: {rd / 'metrics.json'}, {png}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
