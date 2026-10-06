# -*- coding: utf-8 -*-
"""==================== e308 — THE DOUBLETALK PRECURSOR ====================
A CPU-ONLY desk cell (threads <= 4, NO GPU, NO envelope writes, timestamps
datetime.now(UTC) only). A prior executor died on a model failure before any
artifacts; this file is the CLEAN START — nothing from the dead attempt is
reused. This script is committed at birth BEFORE any compute (the frozen
registration below is the birth record).

THE QUESTION (verbatim from the dispatch):
e302 (doublethink) predicts two contradictory maintained facts manifest as
a BIMODAL next-token distribution at the shared masked positions. The
precursor, answerable NOW: in e289's CONTRADICTED-WITH-CONTROLLER final
state (the fact held at 3.6x while the corpus taught the OPPOSITE tokens
at the same positions), is the distribution ALREADY bimodal — both claims
holding — or unimodally the held fact?

PROVENANCE (what this cell builds on — extend, never repeat):
- e289 (CONTROLLER-WINS-THE-TUG): the (b) CONTRADICTED-WITH-CONTROLLER
  post state checkpointed at runs/checkpoints/
  e289_CONTRADICTED-WITH-CONTROLLER_post.pt (post g0 = 0.9568988680839539,
  survival x3.6157 vs the committed 0.26464763283729553); the (a) passive
  corpse state checkpointed too (post g0 = 0.0024622573982924223,
  x0.0093 — below the family dead bar). The contradiction construction:
  the corpus bank IS the original text at the battery's own contexts —
  the 60-window bank (anchor_full), the 7 masked name positions [PRE,
  PRE+7) of EVERY window decode to the ORIGINAL HOST OPENING at that
  position (FLORIZE/ELIZABE; mix FLORIZEL 19 / ELIZABETH 41), the
  pre-contexts BIT-IDENTICAL to the g0 battery's rows; 48 draws/step,
  full-window CE, 400 steps.
- lab/e288_error_gated.py / lab/e289_cons_limits.py: the read/probe
  conventions (G1.battery_cell — p(Z) at the last position of each
  130-token battery row; G1.evl_load — the registered instrument load).
- lab/e283_established_collision.py: the loaded-fact baseline
  conventions (the fact checkpointed, md5/flat-md5 bound, read-measured).
- THINKING.md T271: the interpretation this precursor serves (the
  controller's economic case strongest exactly under contradiction).

WHAT IS NEW: nobody has ever READ THE FULL NEXT-TOKEN DISTRIBUTION at the
contested masked positions. Every read so far collapsed it to mean p(Z)
(the battery). The doubletalk question lives in the SHAPE of the
distribution — does the held fact's mass coexist with the taught opposite,
or exclude it? e308 reads the distribution; e302 will then ask whether two
controllers can each own one mode.

THE PROBE (frozen at birth, BEFORE compute):
- THE POSITIONS: the fact's masked positions in the 60-window bank — the
  7 name-span offsets j = 0..6 (the y-positions predicting x[PRE+j]; the
  corpus's denial-CE trained these positions on the bank windows, the
  controller's name-CE trained the SAME offsets on the install windows).
  PRIMARY := offset 0 (60 positions, >= 20 required): the ONE masked
  position whose input is BIT-IDENTICAL in both trainings (the 130-token
  pre-context — G_CONTRA's precontext_bit_equal_g0_battery) — the shared
  battleground in the strictest sense. SECONDARY (co-reported, both
  disclosed): offsets 1..6 under BOTH prefixes — the bank window's own
  prefix (the corpus's view: the host opening so far) and the install
  window's prefix (the controller's view: ZEPHYRA so far).
- THE READ: the full next-token softmax (vocab 65) at each position, per
  state (baseline / (a) / (b)); extracted per position: P(ZEPHYRA[j])
  ("P_Z"), P(the original host char at that offset) ("P_H"), the top-5
  masses + their chars, the entropy (nats, full vocab), the argmax char,
  and the ranks of the two claim tokens.
- THE BIMODALITY INDEX per state (PRIMARY read):
  frac_both := frac(P_Z >= 0.10 AND P_H >= 0.10) — "both-mass";
  frac_Z := frac(P_Z >= 0.10); frac_H := frac(P_H >= 0.10);
  frac_argZ := frac(argmax == ZEPHYRA[0])  [offset-0 form];
  THE VALLEY TEST (criterion disclosed here): at a position let r_Z, r_H
  be the 1-based descending-mass ranks of the ZEPHYRA token and the host
  token; the valley := the max mass over all tokens ranked STRICTLY
  BETWEEN them (vacuous — trivially passes — when |r_Z - r_H| = 1, i.e.
  adjacent ranks); the position PASSES when both claims >= 0.10 AND the
  valley <= min(P_Z, P_H) — the two claims stand as separated peaks with
  nothing taller between them.

FROZEN BARS (verbatim from the dispatch; adjudicated on the (b) state at
the PRIMARY read; the table reports all three states):
- ALREADY-BIMODAL: (b) holds both claims (P(Z) and P(host) each >= 10%
  at >= 1/3 of positions) — doublethink's signature pre-exists in
  single-fact contradiction; e302 sharpens to: can two controllers each
  own one mode?
- HELD-UNIMODAL: P(Z) dominates broadly — the controller's win is
  exclusive; doublethink genuinely untested until e302.
- SUPPRESSED-BOTH: neither mass >= 10% broadly (the position crushed to
  uncertainty — entropy high): the battleground neither claim holds.
- MIXED: position-dependent — the table verbatim.

THE ADJUDICATION ORDER (frozen): ALREADY-BIMODAL if frac_both >= 1/3 at
the primary (20 of 60) — else HELD-UNIMODAL if frac_Z >= 2/3 AND
frac_argZ >= 1/2 ("dominates broadly", operationalized here and disclosed)
— else SUPPRESSED-BOTH if frac_Z < 1/3 AND frac_H < 1/3 — else MIXED.
The pooled secondary (all 420 masked positions, bank prefix) is
co-reported with the same index; the verdict stands on the PRIMARY.

VERIFICATION (hard gates — a failure halts):
G_BANK: the bank rebuilt from data/input.txt + e043's constants reproduces
  the committed construction EXACTLY (mix FLORIZEL 19 / ELIZABETH 41;
  masked decode = host openings 60/60; ZEPH count 0; pre-contexts
  bit-equal the g0 battery rows; battery shapes 60x[118/130/142]).
G_TOKENS: no offset has ZEPHYRA[j] == host[j] (the two claims never
  collide into one token — P_Z and P_H are distinct masses at every
  position; machine-verified).
G_STATES: the three states loaded and verified BOTH ways — (i) artifact
  md5 vs the committed record where one exists (the fact: 0f6dc1cf4685
  0ce655dfafc9c853d467); (ii) CONTENT: the loaded state's flat-md5 (for
  the fact: bound to e289's G_FACTLOAD ebebb4472725d582dd74928493f1bfb3)
  and the g0/gm12 battery reads REPRODUCED within 5e-4 of the committed
  values (the post states' file md5s recorded fresh — they were not
  committed anywhere before).
G_CPU: threads <= 4, cuda untouched (no tensor ever leaves the CPU), no
  writes to runs/_envelope_log.jsonl.

DISCLOSURES (carried into metrics + REPORT):
- The offset-0 primary is the only masked position where both trainings
  shared the input bit-exactly; offsets 1-6 differ in input between the
  denial stream (host prefix) and the maintenance stream (ZEPHYRA prefix)
  — both prefix variants probed, both disclosed.
- 'P(host)' is the ORIGINAL TEXT's char at that offset (the source's own
  assertion), NOT a synthesized counterfactual; at offset 0 the host char
  is 'F' (19 windows) or 'E' (41).
- n=1 per state, one lineage, one session; the e289 n1 lottery caveat
  carried verbatim; NOTHING guaranteed — the bars cover all branches.
- A pure read cell: NO training, NO writes to any checkpoint, NO state
  mutation; the loaded nets are eval-disarmed exactly as the family's
  reads always do (G1.evl_load).
==========================================================================
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- CPU lock
import torch                                         # noqa: E402

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass                                            # already initialized
assert torch.zeros(1).device.type == "cpu"          # the CPU lock, live

import numpy as np                                   # noqa: E402
import torch.nn.functional as F                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                        # noqa: E402
from common import CharCorpus, save_json             # noqa: E402

import e043_install as E43                           # noqa: E402
import g1b_continuity as GB                          # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                        # noqa: E402

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

# THE CPU LOCK, RE-ASSERTED (fix committed before recompute): g1b/G1 set
# 8 threads at import (their shared-machine convention) and override the
# module-top lock; this cell's dispatch cap is threads <= 4.
torch.set_num_threads(4)
assert torch.get_num_threads() <= 4

# ------------------------------------------------------------------ helpers
def utcnow() -> str:
    """The house timestamp rule: datetime.now(UTC) only, ISO Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(m: str) -> None:
    print(f"[{utcnow()}] {m}", flush=True)


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                # noqa: BLE001
        return "unavailable"


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def flat_md5(net) -> str:
    return hashlib.md5(
        flat_params_cpu(net).numpy().tobytes()).hexdigest()


# ------------------------------------------------------------- frozen binds
REPO = E43.REPO
CKPT = REPO / "runs" / "checkpoints"
RD = REPO / "runs" / "e308"
RD.mkdir(parents=True, exist_ok=True)

FACT_CK = CKPT / "e261_K10K_inst_resume.pt"
POST_A_CK = CKPT / "e289_CONTRADICTED-NO-CONTROLLER_post.pt"
POST_B_CK = CKPT / "e289_CONTRADICTED-WITH-CONTROLLER_post.pt"
E289_METRICS = REPO / "runs" / "e289" / "metrics.json"

# committed literals (the verification spine — from e289's metrics)
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_FLAT_MD5 = "ebebb4472725d582dd74928493f1bfb3"
FACT_G0 = 0.26464763283729553
FACT_GM12 = 0.10525520890951157
POST_A_G0 = 0.0024622573982924223
POST_A_GM12 = 0.0018603801727294922
POST_B_G0 = 0.9568988680839539
POST_B_GM12 = 0.9355171322822571
READ_TOL = 5e-4

NAME = G1.NAME                       # "ZEPHYRA"
PRE = G1.PRE                         # 130
BLOCK = G1.BLOCK                     # 256
MASS_BAR = 0.10                      # the 10% claim bar (frozen)
FRAC_BAR = 1.0 / 3.0                 # the >= 1/3 positions bar (frozen)
TOPK = 5                             # per-position top-k masses recorded

STATES = [
    ("BASELINE-LOADED-FACT", FACT_CK),
    ("A-DENIED-CORPSE", POST_A_CK),
    ("B-HELD-CONTROLLER", POST_B_CK),
]

# ------------------------------------------------------------- the artifact
metrics: dict = {
    "experiment": "e308_doubletalk",
    "phase": ("THE DOUBLETALK PRECURSOR (CPU-only desk cell): the full "
              "next-token distribution at e289's contested masked "
              "positions — baseline vs denied corpse vs held — is the "
              "held state ALREADY bimodal?"),
    "date": utcnow(),
    "status": "PARTIAL (progressive writes; this run started)",
    "registration": {
        "question_verbatim": (
            "e302 (doublethink) predicts two contradictory maintained "
            "facts manifest as a BIMODAL next-token distribution at the "
            "shared masked positions. The precursor, answerable NOW: in "
            "e289's CONTRADICTED-WITH-CONTROLLER final state (the fact "
            "held at 3.6x while the corpus taught the OPPOSITE tokens at "
            "the same positions), is the distribution ALREADY bimodal — "
            "both claims holding — or unimodally the held fact?"),
        "bars_verbatim": {
            "ALREADY-BIMODAL": ("(b) holds both claims (P(Z) and "
                                "P(host) each >= 10% at >= 1/3 of "
                                "positions) — doublethink's signature "
                                "pre-exists in single-fact contradiction; "
                                "e302 sharpens to: can two controllers "
                                "each own one mode?"),
            "HELD-UNIMODAL": ("P(Z) dominates broadly — the controller's "
                              "win is exclusive; doublethink genuinely "
                              "untested until e302."),
            "SUPPRESSED-BOTH": ("neither mass >= 10% broadly (the "
                                "position crushed to uncertainty — "
                                "entropy high): the battleground neither "
                                "claim holds."),
            "MIXED": "position-dependent — the table verbatim.",
        },
        "adjudication_order": ("ALREADY-BIMODAL if frac_both >= 1/3 at "
                               "the primary (>= 20 of 60) — else "
                               "HELD-UNIMODAL if frac_Z >= 2/3 AND "
                               "frac_argZ >= 1/2 — else SUPPRESSED-BOTH "
                               "if frac_Z < 1/3 AND frac_H < 1/3 — else "
                               "MIXED; verdict on the PRIMARY "
                               "(offset-0) read; the pooled-420 "
                               "secondary co-reported"),
        "primary_secondary": (
            "PRIMARY := offset 0 (60 positions; the only masked position "
            "whose input is bit-identical in both trainings — the "
            "130-token pre-context, G_CONTRA's own bind). SECONDARY := "
            "offsets 1-6 under both prefixes (bank/host prefix — the "
            "corpus's view; install/ZEPHYRA prefix — the controller's "
            "view) + the pooled 420-position index."),
        "valley_criterion": (
            "valley := max mass over tokens ranked strictly between the "
            "two claim tokens (vacuous pass when adjacent); a position "
            "passes when both claims >= 0.10 AND valley <= min(P_Z, P_H)"),
        "clean_start": ("a prior executor died on a model failure before "
                        "ANY artifacts; nothing reused from that attempt"),
        "cpu_only": "threads <= 4, no GPU, no envelope writes",
        "timestamp_rule": "datetime.now(UTC) only",
    },
    "gates": {},
    "reads": {},
    "bimodality_index": {},
    "provenance": {},
    "disclosures": {},
}


def write_partial(note: str) -> None:
    metrics["date"] = utcnow()
    metrics["status"] = f"PARTIAL — {note}"
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"[progressive] metrics.json written ({note})")


# ========================================================================
# P0 — the bank rebuilt + gated (G_BANK / G_TOKENS)
# ========================================================================
def phase0() -> dict:
    t0 = time.time()
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)

    G_VOCAB = {"vocab_size": int(corpus.vocab_size),
               "expected": 65,
               "pass": bool(corpus.vocab_size == 65)}
    assert G_VOCAB["pass"], f"vocab drift: {G_VOCAB}"

    zeph_count = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": zeph_count,
                  "pass": bool(zeph_count == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{zeph_count}"

    # ---- install_occ: e289's construction VERBATIM ----------------------
    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = __import__("random").Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    # ---- the battery (g-12/g0/g+12) + the bank + the install windows ----
    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    g0_ids, gm12_ids = bat_ids[0], bat_ids[-12]
    G_BATTERY = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "expected": {"g-12": [60, PRE - 12], "g0": [60, PRE],
                              "g+12": [60, PRE + 12]},
                 "pass": bool(list(bat_ids[-12].shape) == [60, PRE - 12]
                              and list(bat_ids[0].shape) == [60, PRE]
                              and list(bat_ids[12].shape)
                              == [60, PRE + 12])}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    anchor_full = torch.stack(
        [train_ids[p - PRE: p - PRE + BLOCK] for p, _ in install_occ])
    name_ids = corpus.encode(NAME)

    def build_win(p, host):
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])

    # ---- G_BANK: the contradiction construction reproduced EXACTLY ------
    decode7 = ["".join(itos[int(i)] for i in
                       anchor_full[i][PRE: PRE + len(NAME)])
               for i in range(60)]
    host_first7 = [h[:len(NAME)] for _, h in install_occ]
    bank_ok = all(decode7[i] == host_first7[i] for i in range(60))
    pre_ok = all(torch.equal(anchor_full[i][:PRE], g0_ids[i])
                 for i in range(60))
    winname_ok = all("".join(itos[int(i)] for i in
                             win_i[i][PRE: PRE + len(NAME)]) == NAME
                     for i in range(60))
    G_BANK = {
        "form": ("the 60-window ORIGINAL-TEXT bank (anchor_full) + the "
                 "g0 battery + the install windows rebuilt from "
                 "data/input.txt + e043's constants — e289's "
                 "construction reproduced and DECODE-VERIFIED"),
        "install_mix": mix,
        "masked_decode_matches_host_opening_60_of_60": bool(bank_ok),
        "host_openings": sorted(set(decode7)),
        "precontext_bit_equal_g0_battery": bool(pre_ok),
        "install_windows_decode_name_60_of_60": bool(winname_ok),
        "bank_zeph_count": int("".join(
            "".join(itos[int(i)] for i in w) for w in anchor_full
        ).count("ZEPH")),
        "held_mix": {"FLORIZEL": sum(1 for _, h in held_occ
                                     if h == "FLORIZEL"),
                     "ELIZABETH": sum(1 for _, h in held_occ
                                      if h == "ELIZABETH")},
        "pass": bool(bank_ok and pre_ok and winname_ok and mix ==
                     {"FLORIZEL": 19, "ELIZABETH": 41}),
    }
    assert G_BANK["pass"], f"G_BANK failed: {G_BANK}"

    # ---- G_TOKENS: the two claims never collide into one token ----------
    z_chars = [stoi[c] for c in NAME]                     # ZEPHYRA ids
    h_chars = [[stoi[h[j]] for j in range(len(NAME))]
               for _, h in install_occ]                   # per-window ids
    collisions = [j for j in range(len(NAME))
                  if any(z_chars[j] == hc[j] for hc in h_chars)]
    G_TOKENS = {
        "form": ("per offset j, ZEPHYRA[j] vs the window's host char j — "
                 "the two claims must be DISTINCT tokens at every "
                 "position (else P_Z and P_H would be one mass)"),
        "colliding_offsets": collisions,
        "offset0_host_chars": {"F": int(sum(1 for _, h in install_occ
                                            if h[0] == "F")),
                               "E": int(sum(1 for _, h in install_occ
                                            if h[0] == "E"))},
        "pass": bool(len(collisions) == 0),
    }
    assert G_TOKENS["pass"], f"G_TOKENS failed: {G_TOKENS}"

    metrics["gates"]["G_VOCAB"] = G_VOCAB
    metrics["gates"]["G_NAMEFREE"] = G_NAMEFREE
    metrics["gates"]["G_BANK"] = G_BANK
    metrics["gates"]["G_TOKENS"] = G_TOKENS
    log(f"P0 G_BANK: bank + battery + install windows rebuilt; mix "
        f"{mix}; openings {G_BANK['host_openings']}; pre-context "
        f"bit-equal; G_TOKENS: no claim collisions")
    write_partial("P0 bank rebuilt + gated")
    return {"corpus": corpus, "stoi": stoi, "itos": itos,
            "g0_ids": g0_ids, "gm12_ids": gm12_ids,
            "anchor_full": anchor_full, "win_i": win_i,
            "install_occ": install_occ, "z_chars": z_chars,
            "h_chars": h_chars, "zid": stoi["Z"],
            "secs": time.time() - t0}


# ========================================================================
# P1 — the three states loaded + verified BOTH ways (G_STATES)
# ========================================================================
def phase1(p0: dict) -> dict:
    t0 = time.time()
    e289m = json.loads(E289_METRICS.read_text(encoding="utf-8"))
    nets, rec = {}, {}
    binds = [
        ("BASELINE-LOADED-FACT", FACT_CK, FACT_MD5, FACT_FLAT_MD5,
         FACT_G0, FACT_GM12),
        ("A-DENIED-CORPSE", POST_A_CK, None, None, POST_A_G0, POST_A_GM12),
        ("B-HELD-CONTROLLER", POST_B_CK, None, None, POST_B_G0, POST_B_GM12),
    ]
    for tag, ck, md5_bound, flat_bound, g0_bound, gm12_bound in binds:
        file_md5 = md5of(ck)
        art = torch.load(ck, map_location="cpu", weights_only=False)
        sd = {k: v.detach().clone() for k, v in art["model"].items()}
        net = G1.evl_load(sd)                 # the registered instrument
        net.eval()
        fmd5 = flat_md5(net)
        g0 = G1.battery_cell(net, p0["g0_ids"], p0["zid"])
        gm12 = G1.battery_cell(net, p0["gm12_ids"], p0["zid"])
        md5_ok = (md5_bound is None) or (file_md5 == md5_bound)
        flat_ok = (flat_bound is None) or (fmd5 == flat_bound)
        g0_ok = abs(g0["mean_pz"] - g0_bound) < READ_TOL
        gm12_ok = abs(gm12["mean_pz"] - gm12_bound) < READ_TOL
        rec[tag] = {
            "checkpoint": str(ck.relative_to(REPO)).replace("\\", "/"),
            "file_md5": file_md5,
            "file_md5_bound": md5_bound,
            "file_md5_ok": bool(md5_ok),
            "flat_md5": fmd5,
            "flat_md5_bound": flat_bound,
            "flat_md5_ok": bool(flat_ok),
            "meta_experiment": art.get("meta", {}).get("experiment"),
            "g0_read_measured": g0["mean_pz"],
            "g0_read_bound": g0_bound,
            "g0_abs_diff": abs(g0["mean_pz"] - g0_bound),
            "gm12_read_measured": gm12["mean_pz"],
            "gm12_read_bound": gm12_bound,
            "gm12_abs_diff": abs(gm12["mean_pz"] - gm12_bound),
            "g0_frac_argmax_z": g0["frac_argmax_z"],
            "content_ok": bool(g0_ok and gm12_ok),
            "pass": bool(md5_ok and flat_ok and g0_ok and gm12_ok),
        }
        nets[tag] = net
        assert rec[tag]["pass"], f"G_STATES failed for {tag}: {rec[tag]}"
        log(f"P1 {tag}: md5 {file_md5[:8]} flat {fmd5[:8]} g0 "
            f"{g0['mean_pz']:.6f} (bound {g0_bound:.6f}, d="
            f"{rec[tag]['g0_abs_diff']:.2e}) gm12 d="
            f"{rec[tag]['gm12_abs_diff']:.2e}")

    G_STATES = {
        "form": ("the three states verified BOTH ways: artifact md5 (the "
                 "fact additionally flat-md5-bound to e289's G_FACTLOAD) "
                 "+ CONTENT reproduction of the committed g0/gm12 battery "
                 f"reads within {READ_TOL}"),
        "read_tol": READ_TOL,
        "e289_metrics_md5": md5of(E289_METRICS),
        "e289_verdict": e289m["adjudication"]["verdict"],
        "e289_b_clause": ("(b) HOLDS at x3.6157 while (a) DIES at "
                          "x0.0093 — CONTROLLER-WINS-THE-TUG"),
        "states": rec,
        "pass": bool(all(v["pass"] for v in rec.values())),
    }
    metrics["gates"]["G_STATES"] = G_STATES
    write_partial("P1 the three states loaded + verified (md5 + content)")
    nets_out = nets
    return {"nets": nets_out, "e289m": e289m, "secs": time.time() - t0}


# ========================================================================
# P2 — THE PROBE: the full next-token distribution at the masked positions
# ========================================================================
@torch.no_grad()
def position_distributions(net, prefix_batch: torch.Tensor) -> torch.Tensor:
    """Full softmax at the LAST position of each prefix row (the family's
    own read geometry: battery_cell reads p(zid) here; e308 keeps the
    whole distribution). Returns [n, vocab]."""
    net.eval()
    outs = []
    for i in range(0, prefix_batch.shape[0], 30):
        lg, _ = net(prefix_batch[i:i + 30])
        outs.append(F.softmax(lg[:, -1], dim=-1))
    return torch.cat(outs)


def claim_stats(pr: torch.Tensor, zid: int, hid: int,
                itos) -> dict:
    """One position's extracted read: P_Z, P_H, entropy, top-k, ranks,
    argmax, the valley test."""
    p = pr.double()
    pz, ph = float(p[zid]), float(p[hid])
    order = torch.argsort(pr, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, len(order) + 1)
    rz, rh = int(ranks[zid]), int(ranks[hid])
    lo, hi = min(rz, rh), max(rz, rh)
    between = order[lo:hi - 1] if hi - lo > 1 else None
    valley = (float(pr[between].max()) if between is not None
              and between.numel() else 0.0)
    ent = float(-(p * torch.where(p > 0, p.log(),
                                  torch.zeros_like(p))).sum())
    topv, topi = torch.topk(pr, TOPK)
    both = (pz >= MASS_BAR) and (ph >= MASS_BAR)
    return {
        "p_z": pz, "p_h": ph, "entropy": ent,
        "argmax_char": itos[int(pr.argmax())],
        "rank_z": rz, "rank_h": rh,
        "valley": valley,
        "valley_pass": bool(both and valley <= min(pz, ph)),
        "both": bool(both),
        "topk": [{"char": itos[int(i)], "mass": float(v)}
                 for v, i in zip(topv, topi)],
        "top1_mass": float(topv[0]), "top2_mass": float(topv[1]),
    }


def phase2(p0: dict, p1: dict) -> dict:
    t0 = time.time()
    anchor_full, win_i = p0["anchor_full"], p0["win_i"]
    z_chars, h_chars, itos = p0["z_chars"], p0["h_chars"], p0["itos"]
    reads = {}
    for tag, net in p1["nets"].items():
        per = {"bank": {j: [] for j in range(len(NAME))},
               "install": {j: [] for j in range(len(NAME))}}
        for j in range(len(NAME)):
            # bank prefix (the corpus's view: original text so far)
            pre_b = anchor_full[:, :PRE + j].clone()
            pr_b = position_distributions(net, pre_b)
            # install prefix (the controller's view: ZEPHYRA so far)
            pre_i = win_i[:, :PRE + j].clone()
            pr_i = position_distributions(net, pre_i)
            for w in range(60):
                per["bank"][j].append(claim_stats(
                    pr_b[w], z_chars[j], h_chars[w][j], itos))
                per["install"][j].append(claim_stats(
                    pr_i[w], z_chars[j], h_chars[w][j], itos))
        reads[tag] = per
        log(f"P2 {tag}: 7 offsets x 2 prefixes x 60 windows read")

    metrics["reads"]["per_state"] = {
        tag: {"bank": {"offsets": {
                 j: {"positions": v} for j, v in per["bank"].items()},
               },
              "install": {"offsets": {
                 j: {"positions": v} for j, v in per["install"].items()},
                 }}
        for tag, per in reads.items()}
    write_partial("P2 the full distributions read (all states/offsets)")
    return {"reads": reads, "secs": time.time() - t0}


# ========================================================================
# P3 — the bimodality index + the table + adjudication
# ========================================================================
def index_over(positions: list) -> dict:
    n = len(positions)
    pz = [q["p_z"] for q in positions]
    ph = [q["p_h"] for q in positions]
    ent = [q["entropy"] for q in positions]
    t1 = [q["top1_mass"] for q in positions]
    t2 = [q["top2_mass"] for q in positions]
    f_both = sum(q["both"] for q in positions) / n
    f_z = sum(q["p_z"] >= MASS_BAR for q in positions) / n
    f_h = sum(q["p_h"] >= MASS_BAR for q in positions) / n
    f_argz = sum(q["argmax_char"] == NAME[0] for q in positions) / n
    f_argh = sum(q["argmax_char"] in ("F", "E") for q in positions) / n
    f_valley = sum(q["valley_pass"] for q in positions) / n
    return {
        "n_positions": n,
        "mean_p_z": float(np.mean(pz)), "median_p_z": float(np.median(pz)),
        "mean_p_h": float(np.mean(ph)), "median_p_h": float(np.median(ph)),
        "frac_p_z_ge_10pct": f_z, "frac_p_h_ge_10pct": f_h,
        "frac_both_ge_10pct": f_both,
        "frac_argmax_z": f_argz, "frac_argmax_host_first_char": f_argh,
        "frac_valley_pass": f_valley,
        "mean_entropy_nats": float(np.mean(ent)),
        "median_entropy_nats": float(np.median(ent)),
        "mean_top1_mass": float(np.mean(t1)),
        "mean_top2_mass": float(np.mean(t2)),
    }


def per_state_summary(reads: dict) -> dict:
    out = {}
    for tag, per in reads.items():
        prim = per["bank"][0]                       # the PRIMARY read
        pooled_bank, pooled_inst = [], []
        per_off = {"bank": {}, "install": {}}
        for j in range(len(NAME)):
            per_off["bank"][j] = index_over(per["bank"][j])
            per_off["install"][j] = index_over(per["install"][j])
            pooled_bank += per["bank"][j]
            pooled_inst += per["install"][j]
        out[tag] = {
            "primary_offset0_shared": index_over(prim),
            "pooled_420_bank_prefix": index_over(pooled_bank),
            "pooled_420_install_prefix": index_over(pooled_inst),
            "per_offset": per_off,
            "string_level": {
                "p_zephyra_given_install_prefix": float(np.mean([
                    float(np.prod([per["install"][j][w]["p_z"]
                                   for j in range(len(NAME))]))
                    for w in range(60)])),
                "p_host_opening_given_bank_prefix": float(np.mean([
                    float(np.prod([per["bank"][j][w]["p_h"]
                                   for j in range(len(NAME))]))
                    for w in range(60)])),
                "note": ("geometric means over the 60 windows of the "
                         "7-offset string products — the continuation-"
                         "level co-report, not a bar"),
            },
        }
    return out


def adjudicate(idx: dict) -> tuple:
    f_both = idx["frac_both_ge_10pct"]
    f_z = idx["frac_p_z_ge_10pct"]
    f_h = idx["frac_p_h_ge_10pct"]
    f_argz = idx["frac_argmax_z"]
    if f_both >= FRAC_BAR:
        return "ALREADY-BIMODAL", (
            f"frac_both {f_both:.4f} >= 1/3 — both claims (P(Z) and "
            f"P(host) each >= 10%) hold at >= 1/3 of the shared masked "
            f"positions: doublethink's signature PRE-EXISTS in "
            f"single-fact contradiction; e302 sharpens to: can two "
            f"controllers each own one mode?")
    if f_z >= 2 / 3 and f_argz >= 0.5:
        return "HELD-UNIMODAL", (
            f"frac_both {f_both:.4f} < 1/3 while P(Z) >= 10% at "
            f"{f_z:.4f} of positions and argmax=Z at {f_argz:.4f} — "
            f"the controller's win is EXCLUSIVE; doublethink genuinely "
            f"untested until e302")
    if f_z < FRAC_BAR and f_h < FRAC_BAR:
        return "SUPPRESSED-BOTH", (
            f"neither claim >= 10% at >= 1/3 of positions (frac_Z "
            f"{f_z:.4f}, frac_H {f_h:.4f}); mean entropy "
            f"{idx['mean_entropy_nats']:.4f} nats — the battleground "
            f"crushed to uncertainty, neither claim holds")
    return "MIXED", (
        f"position-dependent (frac_Z {f_z:.4f}, frac_H {f_h:.4f}, "
        f"frac_both {f_both:.4f}) — the table verbatim")


def phase3(p2: dict) -> dict:
    t0 = time.time()
    summ = per_state_summary(p2["reads"])
    metrics["bimodality_index"] = summ
    verdict, clause = adjudicate(
        summ["B-HELD-CONTROLLER"]["primary_offset0_shared"])
    sec_v, sec_c = adjudicate(
        summ["B-HELD-CONTROLLER"]["pooled_420_bank_prefix"])
    adjs = {}
    for tag in summ:
        v, c = adjudicate(summ[tag]["primary_offset0_shared"])
        adjs[tag] = {"verdict": v, "clause": c}
    metrics["adjudication"] = {
        "bars_verbatim": metrics["registration"]["bars_verbatim"],
        "order": metrics["registration"]["adjudication_order"],
        "primary": {"read": "B-HELD-CONTROLLER offset-0 shared "
                            "(60 positions)",
                    "verdict": verdict, "clause": clause},
        "secondary_pooled_420_bank_prefix": {"verdict": sec_v,
                                             "clause": sec_c},
        "per_state_at_primary": adjs,
        "note": ("the frozen bars speak about (b); the per-state rows "
                 "are the progression table (baseline -> corpse -> held)"),
    }
    log(f"P3 VERDICT (primary, (b)): {verdict}")
    log(f"P3 clause: {clause}")
    log(f"P3 secondary (pooled-420, bank prefix, (b)): {sec_v}")
    write_partial("P3 index + adjudication")
    return {"summary": summ, "verdict": verdict, "clause": clause,
            "secondary_verdict": sec_v, "secs": time.time() - t0}


# ========================================================================
# P4 — the figure + REPORT.md
# ========================================================================
def make_table_md(summ: dict, adjs: dict) -> str:
    rows = ["| state | mean P_Z | mean P_H | frac Z>=10% | frac H>=10% "
            "| frac BOTH | frac argZ | frac valley | mean H (nats) | "
            "mean top1 | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["BASELINE-LOADED-FACT", "A-DENIED-CORPSE",
             "B-HELD-CONTROLLER"]
    for tag in order:
        s = summ[tag]["primary_offset0_shared"]
        rows.append(
            f"| {tag} | {s['mean_p_z']:.4f} | {s['mean_p_h']:.4f} | "
            f"{s['frac_p_z_ge_10pct']:.3f} | {s['frac_p_h_ge_10pct']:.3f} "
            f"| {s['frac_both_ge_10pct']:.3f} | "
            f"{s['frac_argmax_z']:.3f} | {s['frac_valley_pass']:.3f} | "
            f"{s['mean_entropy_nats']:.4f} | {s['mean_top1_mass']:.4f} "
            f"| {adjs[tag]['verdict']} |")
    return "\n".join(rows)


def phase4(p2: dict, p3: dict) -> None:
    t0 = time.time()
    reads = p2["reads"]
    summ = p3["summary"]
    adjs = metrics["adjudication"]["per_state_at_primary"]
    order = ["BASELINE-LOADED-FACT", "A-DENIED-CORPSE",
             "B-HELD-CONTROLLER"]
    short = {"BASELINE-LOADED-FACT": "baseline (loaded fact)",
             "A-DENIED-CORPSE": "(a) denied corpse",
             "B-HELD-CONTROLLER": "(b) held (controller)"}
    colors = {"BASELINE-LOADED-FACT": "#7f7f7f",
              "A-DENIED-CORPSE": "#d62728",
              "B-HELD-CONTROLLER": "#1f77b4"}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    fig.suptitle("e308 THE DOUBLETALK PRECURSOR — the full next-token "
                 "distribution at e289's contested masked positions "
                 "(CPU desk cell)", fontsize=12)

    # (0,0) sorted per-position masses at the primary (offset 0)
    ax = axes[0][0]
    for tag in order:
        pos = reads[tag]["bank"][0]
        pz = sorted((q["p_z"] for q in pos), reverse=True)
        ph = sorted((q["p_h"] for q in pos), reverse=True)
        ax.plot(pz, color=colors[tag], lw=2.2,
                label=f"{short[tag]}: P(Z)")
        ax.plot(ph, color=colors[tag], lw=1.4, ls="--",
                label=f"{short[tag]}: P(host)")
    ax.axhline(0.10, color="k", lw=0.8, ls=":", label="the 10% claim bar")
    ax.set_title("PRIMARY — offset-0 shared masked positions (60): "
                 "sorted claim masses")
    ax.set_xlabel("position rank (sorted)")
    ax.set_ylabel("mass")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, ncol=2)

    # (0,1) mean per-token mass, top tokens aggregated across the 60
    # primary positions (from the per-position top-5 reads)
    ax = axes[0][1]
    width = 0.27
    chars_seen: dict = {}
    for tag in order:
        pos = reads[tag]["bank"][0]
        agg: dict = {}
        for q in pos:
            for e in q["topk"]:
                agg[e["char"]] = agg.get(e["char"], 0.0) + e["mass"]
        for c, v in agg.items():
            chars_seen.setdefault(c, {})[tag] = v / 60.0
    ranked = sorted(chars_seen.items(),
                    key=lambda kv: -max(kv[1].values()))[:9]
    xs = np.arange(len(ranked))
    for k, tag in enumerate(order):
        ax.bar(xs + (k - 1) * width,
               [chars_seen[c].get(tag, 0.0) for c, _ in ranked],
               width=width, color=colors[tag],
               label=short[tag])
    ax.set_xticks(xs)
    ax.set_xticklabels([repr(c) for c, _ in ranked], fontsize=9)
    ax.set_title("PRIMARY — mean per-token mass (top tokens, top-5 "
                 "aggregated over 60 positions)")
    ax.set_ylabel("mean mass")
    ax.legend(fontsize=8)

    # (1,0) per-offset mean claim masses, bank vs install prefix, (b)
    ax = axes[1][0]
    tag = "B-HELD-CONTROLLER"
    js = list(range(len(NAME)))
    mb = [np.mean([q["p_z"] for q in reads[tag]["bank"][j]])
          for j in js]
    hb = [np.mean([q["p_h"] for q in reads[tag]["bank"][j]])
          for j in js]
    mi = [np.mean([q["p_z"] for q in reads[tag]["install"][j]])
          for j in js]
    hi = [np.mean([q["p_h"] for q in reads[tag]["install"][j]])
          for j in js]
    ax.plot(js, mb, "o-", color="#1f77b4", lw=2,
            label="(b) P(ZEPHYRA[j]) — bank/host prefix")
    ax.plot(js, hb, "o--", color="#d62728", lw=2,
            label="(b) P(host[j]) — bank/host prefix")
    ax.plot(js, mi, "s-", color="#9467bd", lw=1.6,
            label="(b) P(ZEPHYRA[j]) — install/ZEPHYRA prefix")
    ax.plot(js, hi, "s--", color="#ff7f0e", lw=1.6,
            label="(b) P(host[j]) — install/ZEPHYRA prefix")
    ax.axhline(0.10, color="k", lw=0.8, ls=":")
    ax.set_title("SECONDARY — (b) the 7 masked offsets under BOTH "
                 "prefixes")
    ax.set_xlabel("name-span offset j (0 = the shared position)")
    ax.set_ylabel("mean mass")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)

    # (1,1) entropy + top1 vs top2 across offsets per state (bank prefix)
    ax = axes[1][1]
    for tag in order:
        ent = [np.mean([q["entropy"] for q in reads[tag]["bank"][j]])
               for j in js]
        ax.plot(js, ent, "o-", color=colors[tag], lw=2,
                label=f"{short[tag]}: entropy")
    ax.set_title("SECONDARY — mean entropy per offset (bank prefix)")
    ax.set_xlabel("name-span offset j")
    ax.set_ylabel("mean entropy (nats)")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = RD / "e308_doubletalk.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    log(f"P4 figure -> {png.name}")

    table = make_table_md(summ, adjs)
    b = summ["B-HELD-CONTROLLER"]
    rep = [
        "# e308 — THE DOUBLETALK PRECURSOR (CPU desk cell)",
        "",
        f"**Date:** {utcnow()}  ",
        f"**Verdict (PRIMARY, (b) at the offset-0 shared masked "
        f"positions): {p3['verdict']}**",
        "",
        p3["clause"],
        "",
        f"Secondary co-report (pooled 420 masked positions, bank "
        f"prefix, (b)): **{p3['secondary_verdict']}**",
        "",
        "## The progression table (PRIMARY: offset-0, the shared "
        "masked position, 60 positions)",
        "",
        table,
        "",
        "## (b) the index in full",
        "",
        "- offset-0 shared (primary): "
        f"P_Z {b['primary_offset0_shared']['mean_p_z']:.4f}, P_H "
        f"{b['primary_offset0_shared']['mean_p_h']:.4f}, frac_both "
        f"{b['primary_offset0_shared']['frac_both_ge_10pct']:.3f}, "
        f"frac_valley "
        f"{b['primary_offset0_shared']['frac_valley_pass']:.3f}",
        "- pooled-420 bank prefix: P_Z "
        f"{b['pooled_420_bank_prefix']['mean_p_z']:.4f}, P_H "
        f"{b['pooled_420_bank_prefix']['mean_p_h']:.4f}, frac_both "
        f"{b['pooled_420_bank_prefix']['frac_both_ge_10pct']:.3f}",
        "- pooled-420 install prefix: P_Z "
        f"{b['pooled_420_install_prefix']['mean_p_z']:.4f}, P_H "
        f"{b['pooled_420_install_prefix']['mean_p_h']:.4f}, frac_both "
        f"{b['pooled_420_install_prefix']['frac_both_ge_10pct']:.3f}",
        "- string-level: P('ZEPHYRA' | install prefix) "
        f"{b['string_level']['p_zephyra_given_install_prefix']:.3e}; "
        f"P(host opening | bank prefix) "
        f"{b['string_level']['p_host_opening_given_bank_prefix']:.3e}",
        "",
        "## What was done",
        "",
        "The three e289-lineage states (the loaded-fact baseline, the "
        "(a) denied corpse, the (b) held state) loaded read-only and "
        "verified md5 + content (the committed g0/gm12 reads reproduced "
        f"within {READ_TOL}). At the fact's masked positions in the "
        "60-window bank — the 7 name-span offsets, PRIMARY = offset 0, "
        "the only position whose input is bit-identical in both e289 "
        "trainings — the FULL next-token distribution read per state "
        "under both prefixes (bank/host = the corpus's view; "
        "install/ZEPHYRA = the controller's view); per position: the "
        "two claim masses, top-5 masses, entropy, ranks, and the "
        "valley test.",
        "",
        "## Disclosures",
        "",
        "- CPU-only desk cell (threads<=4, no GPU, no envelope writes); "
        "a pure read — no training, no checkpoint mutation; timestamps "
        "datetime.now(UTC) only.",
        "- Offset 0 is the only masked position where the denial "
        "stream and the maintenance stream shared the input "
        "bit-exactly (the 130-token pre-context); offsets 1-6 differ "
        "in input between the trainings — both prefix variants "
        "probed and co-reported.",
        "- 'P(host)' = the ORIGINAL TEXT's char at that offset "
        "(FLORIZE/ELIZABE openings; F x19, E x41 at offset 0) — the "
        "source's own assertion, not a synthesized counterfactual; "
        "G_TOKENS verified the two claims never collide into one "
        "token at any offset.",
        f"- Valley criterion (disclosed): between the two claim ranks, "
        f"no token taller than min(P_Z, P_H); vacuous pass when the "
        f"claims are rank-adjacent.",
        "- n=1 per state, one lineage, one session (the e289 lottery "
        "caveat carried); nothing guaranteed; the bars cover all "
        "branches.",
        "",
        "Artifacts: metrics.json (full per-position reads), "
        "e308_doubletalk.png, this REPORT.md.",
        "",
    ]
    (RD / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    log("P4 REPORT.md written")


# ========================================================================
def main() -> None:
    log("=" * 78)
    log("e308 THE DOUBLETALK PRECURSOR — clean start (CPU-only desk cell)")
    log("=" * 78)
    metrics["provenance"]["git_head_at_start"] = git_head()
    metrics["provenance"]["script"] = str(Path(__file__).resolve())
    metrics["provenance"]["machinery_imported_from"] = [
        str(Path(GB.__file__).resolve()),
        str(Path(G1.__file__).resolve()),
        str(Path(E43.__file__).resolve()),
    ]
    metrics["provenance"]["timestamps"] = "datetime.now(UTC) only"
    metrics["provenance"]["threads"] = int(torch.get_num_threads())
    metrics["provenance"]["cuda_used"] = False
    write_partial("birth record (pre-P0)")

    p0 = phase0()
    p1 = phase1(p0)
    p2 = phase2(p0, p1)
    p3 = phase3(p2)
    phase4(p2, p3)

    metrics["disclosures"] = {
        "the_position_choice": (
            "PRIMARY = offset 0 (the shared masked position — input "
            "bit-identical in both trainings); SECONDARY = offsets 1-6 "
            "under both prefixes + pooled-420; the verdict stands on "
            "the primary, everything co-reported"),
        "the_host_read": ("P(host) is the original text's own char at "
                          "that offset (decode-verified bank; FLORIZE/"
                          "ELIZABE), never synthesized"),
        "the_valley": ("max mass strictly between the two claim ranks "
                       "<= min(P_Z, P_H); vacuous when rank-adjacent"),
        "n1_caveat": ("n=1 per state, one lineage, one session (the "
                      "g-series standing lottery note carried verbatim "
                      "from e289); nothing guaranteed"),
        "pure_read": ("no training, no state mutation, no envelope "
                      "writes; the loaded nets eval-disarmed via "
                      "G1.evl_load exactly as the family's reads"),
        "clean_start": ("the prior executor died before ANY artifacts; "
                        "this cell began from zero and is committed at "
                        "birth BEFORE compute"),
    }
    metrics["provenance"]["phase_secs"] = {
        "P0_bank": p0["secs"], "P1_states": p1["secs"],
        "P2_probe": p2["secs"], "P3_index": p3["secs"]}
    metrics["outputs"] = [
        str((RD / "metrics.json").resolve()),
        str((RD / "e308_doubletalk.png").resolve()),
        str((RD / "REPORT.md").resolve()),
    ]
    metrics["date"] = utcnow()
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces "
                         "all PARTIAL progressive writes)")
    metrics["phase_note"] = "P4 DONE (probe + index + figure + report)"
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log("=" * 78)
    log(f"e308 COMPLETE — VERDICT: {p3['verdict']}")
    log("=" * 78)


if __name__ == "__main__":
    main()
