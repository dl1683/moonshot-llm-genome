"""E081 — RIF probe: does READING one installed-family name suppress its
coordinate neighbor? (proposal P-A, scratch/explorations_harvest_20260926.md;
operator commission 2026-09-25.)

REGISTERED BEFORE COMPUTE. Eval-only, CPU-only, NO training, NO weight edits:
the entire "exposure" is forward passes.

PREMISE (honesty correction, registered). The operator's brief says "the
install family has TWO names (FLORIZEL and ELIZABETH spliced into the same
windows)". The actual e043/e048 protocol spliced ONE made-up name (ZEPHYRA)
into 90 windows that originally hosted ONE of TWO incumbent names —
FLORIZEL (19 of install-60) or ELIZABETH (41). So the family's two names are
the incumbent HOST pair; they are the two live competitors at the shared
name-onset address. Feasibility probe on e048_dose (2026-09-25, pre-run):
at install-family name slots the distribution is a two-horse race — EL-host
windows: p(E)=0.500 vs p(Z)=0.498; FL-host windows: p(F)=0.544 vs p(Z)=0.454
— i.e. the incumbents and the installed fact compete for the SAME slot.
"Neighbor" is therefore operationalized as the memo intends: a COORDINATE /
POSITIONAL neighbor — the other name occupying the same name-onset slot
(prediction at wpe row 129 after the canonical 130-char install-family
context). The two names' windows are disjoint by construction (a window's
context precedes exactly one host name); we measure cross-name suppression at
matched onset geometry, not same-window competition. DOCUMENTED.

MECHANISM (registered). TinyGPT is stateless: prompt-only exposure can act
ONLY through causal attention within one 256-token session — that IS the
"KV-cache carryover" (the probe attends to the read prefix's keys/values).
Separate forward passes cannot interact (immediate-sequential-evaluation
without a shared window is trivially inert, so the single-session layout is
the only feasible reading manipulation at this scale).

SESSION LAYOUT (uniform across all shifted arms):
  [prefix: 112 tokens] ["\n\n" separator, 2 tokens] [probe context: 130
  tokens] [probe name: 7-9 tokens]   ->  length 251-253 <= 256
  - READ prefix  = 2 elicitation windows x 56 tokens, each = real preceding
    corpus context + the READ name spelled out:
      FLORIZEL read: train_text[p-48:p] + "FLORIZEL" (2 name reads/session)
      ELIZABETH read: train_text[p-47:p] + "ELIZABETH"
      ZEPHYRA read  : train_text[p-49:p] + "ZEPHYRA"  (the installed fact)
  - SHAM prefix  = ONE 112-token corpus segment containing neither target
    name (word-boundary checked), length+charset matched.
  - separator "\n\n" in EVERY shifted arm (read and sham alike) kills the
    direct-junction bigram confound.
  - Probe readout row = 243 (the last context char): p(first name char) at
    the name onset — the exact analogue of the lab's p(Z)@129 battery
    readout, displaced by the 114-token prefix.
  - BASELINE arms ("no exposure") run the probe at its CANONICAL geometry
    (readout row 129, no prefix) — these are instrument-validity references
    (Z gate below) and quantify the shift cost; they are NOT the RIF null
    (see deviation 2).

ARMS.
  PRIMARY (operator's four):
    (a) read-FLORIZEL  -> probe-ELIZABETH (cross)
    (b) sham           -> probe-ELIZABETH (matched-position null)
    (c) read-ELIZABETH -> probe-FLORIZEL  (cross, symmetry)
    (d) canonical baselines (probe-FL / probe-EL / probe-Z @ row 129)
  SECONDARY (flagged, non-gating; self + installed-fact legs):
    self: read-FL -> probe-FL; read-EL -> probe-EL (facilitation?)
    Z legs (memo's fact-A reading): read-Z -> probe-Z; read-FL -> probe-Z;
    read-EL -> probe-Z; read-Z -> probe-FL; read-Z -> probe-EL.
  PROBE BATTERIES (fixed): FL = 29 family FL-host windows (install-60 + held-
  30), EL = 61 family EL-host windows, Z = the 60 install windows (spliced).
  ELICITATION POOLS: occurrences of the read name disjoint from the probe
  battery's occurrences when available (FL pool 45, EL pool 105; Z pool = the
  install windows themselves — for probe-Z the Z-read pool necessarily
  overlaps the probe battery, flagged). B=32 resamples of the elicitation
  set (2 draws with replacement per session; sham segment redrawn), frozen
  RNG.

REGISTERED BARS (feasible version; deviation 2 documents why the operator's
literal "<0.01 sham-vs-no-exposure" gate cannot exist):
  - RIF present iff, in at least one cross direction,
      supp = mean_w [ p(sham)_w - p(read)_w ]  >= 0.05 absolute
    AND its per-window 95% CI (mean +/- 1.96 SEM over probe windows)
    excludes 0, AND the sham-noise gate passes: |placebo split| < 0.01 for
    both probes (sham resamples b<16 vs b>=16, same windows — the resampling
    noise floor of the null arm).
  - Symmetric-in-at-least-one-direction = fires for (a) or (c); both
    directions reported separately.
  - Honest NULL if nothing fires: "reads are pure" is itself the P-A answer.
  Secondary legs reported with the same statistics but never gate.

DEVIATIONS / INFEASIBILITY NOTES (registered before compute):
  1. Stateless transformer: "prompt-only exposure" exists only as
     within-session causal attention (see MECHANISM). The operator's
     alternative "immediate sequential evaluation" across separate passes is
     informationally a no-op and was not run as an arm.
  2. The literal sham gate (|sham - no-exposure| < 0.01) is INFEASIBLE:
     there is no "no-exposure" session at readout row 243 — reaching that
     row requires 114 preceding tokens by construction. Mechanics probe
     (pre-run): the shift itself moves p(E) 0.525 -> 0.736 and collapses the
     install p(Z) -> 0.015 (knife-edge, consistent with T032/e068). The RIF
     statistic is therefore the CONTENT effect at matched position/length
     (read-prefix vs sham-prefix, both @243, per-window paired), and the
     sham gate is re-scoped to the placebo-split noise floor (<0.01).
  3. The Z-probe legs sit at the collapsed geometry (p(Z)@243 ~ 0.015):
     suppression has almost no dynamic range there; facilitation (induction
     from the read prefix) is the live secondary question. Reported, never
     gated.
  4. The memo's "64 elicitation prompts" become 2 name-reads per session
     (the 256-token block fits 112 prefix + canonical 130-char probe + name)
     x B=32 elicitation resamples x the fixed battery — a session-capacity
     constraint, documented.
  5. Z legs go beyond the operator's four arms (the memo's fact-A is the
     ZEPHYRA install; without a Z leg the experiment never reads the
     installed fact). Flagged secondary.
  6. Two checkpoints (operator: "the two name-install checkpoints"):
     e048_dose.pt (primary, the memo's net) and e048_repro.pt (n=2 leg).
  7. (added at smoke, before the full compute; runtime not outcomes)
     Secondary/self/Z legs run at B=16 resamples (primary cross + sham +
     baselines keep B=32); canonical baselines computed once (prefix-free
     sessions are identical across resamples) and tiled.
  8. (added at smoke, before the full compute; honesty reflex) STRING-
     SCRAMBLE control legs: read an anagram of the name — same length,
     charset and unigram inventory, name and probe-opening bigrams destroyed
     (FLORIZEL -> "LFEORZIL", ELIZABETH -> "ZIBLETHEA"; neither contains
     "FL" or "EL") — then probe the other name. If the cross-name effect
     survives the scramble it is fact/char-level; if it vanishes it is
     bigram/induction-level priming of the probe name's opening, NOT
     retrieval-induced forgetting. Non-gating, reported beside the primary.
     Also added at smoke: clustered bootstrap CIs (segments AND windows
     resampled, 2000 draws) and logit-scale deltas for every leg — the
     FL battery is heterogeneous and the raw t-CI is window-variance
     dominated.

READOUTS. Primary: onset probability p(first name char | session) at the
probe readout row. Secondary: teacher-forced joint probability of the full
name and mean name-char logprob. Instrument gate: canonical baseline
p(Z)@129 on install-60 must reproduce the published battery values
(e048_dose 0.5013645 / e048_repro 0.5563087, |drift| < 5e-3 — the e067/e068/
e078 protocol-identity check). No-weight-edit check: wpe/wte bitwise
unchanged after all cells.

Outputs: runs/e081/{metrics.json, rif_probe.png}.
Run: python lab/e081_rif_probe.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                    # CPU experiment

import random as _random                                    # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 8))

import common                                               # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                                  # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- registered constants ------------------------------------------------
PRE = 130                    # canonical probe context length
PREFIX = 112                 # elicitation/sham prefix length (2 x 56)
SEP = "\n\n"                 # uniform junction separator (2 tokens)
ROW_SHIFT = PREFIX + len(SEP) + PRE - 1     # 243: probe readout row, shifted
ROW_CANON = PRE - 1                          # 129: canonical readout row
B_RESAMPLE = 32              # elicitation-set resamples (primary arms)
B_SECONDARY = 16             # secondary/self/Z/scramble legs (deviation 7)
SUPP_BAR = 0.05              # absolute suppression bar (operator)
SHAM_NOISE_BAR = 0.01        # placebo-split noise-floor bar (re-scoped gate)
CI_Z = 1.96
N_BOOT = 2000                # clustered bootstrap draws (deviation 8)
CKPTS = {"e048_dose": "e048_dose.pt", "e048_repro": "e048_repro.pt"}
Z_GATE = {"e048_dose": 0.5013645460208257,     # e067 replication (e078 gate)
          "e048_repro": 0.5563087304433186}    # e068 published
NAME1, NAME2, NAMEZ = "FLORIZEL", "ELIZABETH", "ZEPHYRA"
# deviation 8: anagram controls — same unigrams, no "FL"/"EL" bigrams
SCRAMBLES = {"FL": "LFEORZIL", "EL": "ZIBLETHEA"}

REGISTERED = {
    "question": ("does prompt-only READING of one installed-family name "
                 "(forward pass only, no training, no weight edits) suppress "
                 "its coordinate neighbor's expression at the shared "
                 "name-onset slot?"),
    "neighbor_definition": ("coordinate/positional: the other incumbent name "
                            "at the same name-onset address (130-char "
                            "install-family context, prediction at wpe row "
                            "129 canonical / 243 shifted); windows are "
                            "disjoint by host — cross-name suppression at "
                            "matched geometry is what is measured"),
    "mechanism": ("stateless net: exposure acts only via causal attention "
                  "within one 256-token session (the KV carryover); layout "
                  "[prefix 112]['\\n\\n'][ctx 130][name], readout at row 243"),
    "read_prefix": {"FLORIZEL": "2 x (48-char real ctx + FLORIZEL)",
                    "ELIZABETH": "2 x (47-char real ctx + ELIZABETH)",
                    "ZEPHYRA": "2 x (49-char real ctx + ZEPHYRA)"},
    "sham": "one 112-token corpus segment containing neither target name",
    "arms_primary": ["(a) read-FL -> probe-EL", "(b) sham -> probe-EL",
                     "(c) read-EL -> probe-FL", "(d) canonical baselines"],
    "arms_secondary_flagged": ["self: read-FL->probe-FL, read-EL->probe-EL",
                               "Z legs: read-Z->{probe-Z,probe-FL,probe-EL}, "
                               "read-FL->probe-Z, read-EL->probe-Z",
                               "scramble controls (deviation 8): read-"
                               "LFEORZIL->probe-EL, read-ZIBLETHEA->probe-FL"],
    "batteries": {"FL": "29 family FL-host windows (install+held)",
                  "EL": "61 family EL-host windows (install+held)",
                  "Z": "60 install windows (spliced)"},
    "resamples": f"B={B_RESAMPLE} elicitation-set resamples, frozen RNG",
    "readout": ("primary: onset p(first name char) at probe row; secondary: "
                "teacher-forced joint name probability + mean name-char "
                "logprob"),
    "bars": {
        "rif": ("supp = mean_w[p(sham)_w - p(read)_w] >= 0.05 absolute in "
                ">= 1 cross direction, per-window 95% CI excluding 0, AND "
                "placebo-split |delta| < 0.01 for both probes"),
        "null": ("nothing fires => reads are pure at this scale — the clean "
                 "null is the P-A answer, reported as such"),
        "secondary_non_gating": True,
    },
    "deviations": [
        "1. stateless net: within-session attention is the only exposure "
        "channel; separate-pass 'sequential evaluation' is a no-op, not run",
        "2. operator's literal sham gate infeasible (no no-exposure session "
        "exists at row 243); RIF stat = content effect read-vs-sham at "
        "matched position/length; sham gate re-scoped to placebo-split "
        "noise floor < 0.01; shift cost (sham@243 - base@129) reported "
        "honestly, not gated",
        "3. Z probes sit at the collapsed geometry (p(Z)@243 ~ 0.015, "
        "knife-edge) — suppression has no dynamic range there; Z legs are "
        "facilitation/induction readouts, never gated",
        "4. '64 elicitation prompts' -> 2 name-reads/session (block-size "
        "capacity) x B=32 resamples x fixed battery",
        "5. Z legs beyond the operator's four arms (memo's fact-A is the "
        "install); flagged secondary",
        "6. both install checkpoints run (dose primary, repro n=2)",
        "7. secondary/self/Z/scramble legs at B=16 (primary stays B=32); "
        "canonical baselines computed once and tiled (identical sessions)",
        "8. scramble-anagram control legs + clustered bootstrap CIs + logit-"
        "scale deltas added at smoke, before the full compute (honesty "
        "reflex: discriminate bigram priming from fact-level suppression)",
    ],
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


def load(path: Path):
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def stats(x):
    a = np.asarray(x, dtype=np.float64)
    sem = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem,
            "ci95": [float(a.mean() - CI_Z * sem), float(a.mean() + CI_Z * sem)]}


# ------------------------------------------------------------------ protocol
def rebuild_protocol():
    """Exact e055/e066/e066b/e067/e068/e078 rebuild (verbatim)."""
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    host_occ = []
    for host in (NAME1, NAME2):
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {NAME1: sum(1 for _, h in install_occ if h == NAME1),
           NAME2: sum(1 for _, h in install_occ if h == NAME2)}
    assert mix == {NAME1: 19, NAME2: 41}, f"splice drift {mix}"
    fam = install_occ + held_occ
    probes = {
        "FL": [(p, h) for p, h in fam if h == NAME1],
        "EL": [(p, h) for p, h in fam if h == NAME2],
        "Z": [(p, NAMEZ) for p, _ in install_occ],      # spliced battery
    }
    held_mix = {NAME1: sum(1 for _, h in held_occ if h == NAME1),
                NAME2: sum(1 for _, h in held_occ if h == NAME2)}
    return corpus, train_text, install_occ, probes, mix, held_mix


def build_pools(train_text, install_occ, probes):
    """Elicitation occurrence pools: read-name occurrences disjoint from the
    probe battery when available (fallback flagged). Returns dict keyed by
    (read, probe) -> list[(p, spliced_name)]."""
    fam_fl = {p for p, _ in probes["FL"]}
    fam_el = {p for p, _ in probes["EL"]}
    inst_fl = [(p, NAMEZ) for p, h in install_occ if h == NAME1]
    inst_el = [(p, NAMEZ) for p, h in install_occ if h == NAME2]
    fl_all = [(p, NAME1) for p in E43.find_occ(train_text, NAME1) if p >= 280]
    el_all = [(p, NAME2) for p in E43.find_occ(train_text, NAME2) if p >= 280]
    pools, flags = {}, {}
    for read, probe in (("FL", "EL"), ("FL", "FL"), ("EL", "FL"), ("EL", "EL")):
        bat = fam_fl if probe == "FL" else fam_el
        src = fl_all if read == "FL" else el_all
        pool = [(p, n) for p, n in src if p not in bat]
        if len(pool) < 8:                       # fallback (not expected)
            pool = src
            flags[f"read{read}__probe{probe}"] = "pool fallback to all occs"
        pools[(read, probe)] = pool
    # Z reads: the install windows; for FL/EL probes restrict to the OTHER
    # host so occurrences stay disjoint from the probe battery.
    pools[("Z", "FL")] = inst_el
    pools[("Z", "EL")] = inst_fl
    pools[("Z", "Z")] = [(p, NAMEZ) for p, _ in install_occ]   # overlap flagged
    # FL/EL reads against the Z probe (install-60 battery): exclude the
    # battery's own host occurrences (FL battery-hosts are FL occurrences).
    inst_pos = {p for p, _ in install_occ}
    pools[("FL", "Z")] = [(p, n) for p, n in fl_all if p not in inst_pos]
    pools[("EL", "Z")] = [(p, n) for p, n in el_all if p not in inst_pos]
    flags["readZ__probeZ"] = ("Z-read pool overlaps the Z probe battery "
                              "(ZEPHYRA exists nowhere else) — self leg, "
                              "flagged per deviation 5")
    return pools, flags


# ------------------------------------------------------------------ sessions
def elicitation_prefix(train_text, corpus, read, pool, rng):
    """READ prefix: 2 x 56-token windows (real ctx + the read name), or SHAM:
    one 112-token corpus segment free of both target names."""
    if read == "sham":
        for _ in range(500):
            s = rng.randrange(280, len(train_text) - PREFIX - 2)
            seg = train_text[s: s + PREFIX]
            if (E43.find_occ(seg, NAME1) == [] and E43.find_occ(seg, NAME2) == []
                    and len(seg) == PREFIX):
                return corpus.encode(seg)
        raise RuntimeError("sham sampling failed")
    if read.startswith("scr"):                     # deviation 8 anagram read
        name = SCRAMBLES[read[3:]]
    else:
        name = {"FL": NAME1, "EL": NAME2, "Z": NAMEZ}[read]
    ctx_len = 56 - len(name)
    parts = []
    for _ in range(2):
        p, _host = pool[rng.randrange(len(pool))]
        parts.append(corpus.encode(train_text[p - ctx_len: p] + name))
    ids = torch.cat(parts)
    assert ids.shape[0] == PREFIX, ids.shape
    return ids


@torch.no_grad()
def run_cell(net, corpus, train_text, read, probe_name, probe_occ, pool,
             rng, shifted=True, bs=256, n_res=None):
    """One (read-arm x probe-battery) cell: returns per-(resample, window)
    onset probs and joint probs. rng: the cell's resampling generator.
    Unshifted cells have no prefix -> identical sessions across resamples:
    ONE pass, tiled (deviation 7)."""
    if n_res is None:
        n_res = B_RESAMPLE
    name = {"FL": NAME1, "EL": NAME2, "Z": NAMEZ}[probe_name]
    name_ids = [corpus.stoi[c] for c in name]
    row = ROW_SHIFT if shifted else ROW_CANON
    ctxs = [train_text[p - PRE: p] for p, _ in probe_occ]
    ctx_ids = [corpus.encode(c) for c in ctxs]
    sep_ids = corpus.encode(SEP)
    n_pass = n_res if shifted else 1
    P = np.zeros((n_res, len(probe_occ)), dtype=np.float64)
    J = np.zeros((n_res, len(probe_occ)), dtype=np.float64)
    L = float("inf")
    for b in range(n_pass):
        if shifted:
            pre = elicitation_prefix(train_text, corpus, read, pool, rng)
            sess = [torch.cat([pre, sep_ids, c, corpus.encode(name)])
                    for c in ctx_ids]
        else:
            sess = [torch.cat([c, corpus.encode(name)]) for c in ctx_ids]
        x = torch.stack(sess)
        L = x.shape[1]
        assert L <= net.cfg.block_size, L
        probs_on, probs_j = [], []
        for i in range(0, x.shape[0], bs):
            lg, _ = net(x[i: i + bs])
            pr = F.softmax(lg[:, row: row + len(name), :], -1)   # (n, L, V)
            tgt = torch.tensor(name_ids).view(1, -1, 1).expand(pr.shape[0], -1, 1)
            pnm = torch.gather(pr, 2, tgt).squeeze(-1)           # (n, L)
            probs_on += pnm[:, 0].tolist()
            probs_j += pnm.prod(-1).tolist()
        P[b] = probs_on
        J[b] = probs_j
    if not shifted:
        P[:] = P[0]
        J[:] = J[0]
    return {"P": P, "J": J, "sess_len": int(L), "row": row,
            "read": read, "probe": probe_name}


def arm_stats(cell):
    P, J = cell["P"], cell["J"]
    res_means = P.mean(axis=1)
    return {"onset": stats(P.flatten()),
            "onset_resample_sem": float(res_means.std(ddof=1)
                                        / np.sqrt(len(res_means))),
            "joint_mean": float(J.mean()),
            "joint_log10_sem": float(np.log10(np.maximum(J.flatten(), 1e-300)
                                              ).std(ddof=1)
                                     / np.sqrt(P.size)),
            "n_sessions": int(P.size)}


def delta_cell(sham_cell, read_cell, seed=4242):
    """Per-window paired content effect: sham - read (positive = read
    SUPPRESSED the probe). Includes clustered bootstrap CI (segments AND
    windows resampled, deviation 8) and logit-scale delta."""
    d_on = sham_cell["P"].mean(axis=0) - read_cell["P"].mean(axis=0)
    d_j = np.log10(np.maximum(sham_cell["J"].mean(axis=0), 1e-300)) \
        - np.log10(np.maximum(read_cell["J"].mean(axis=0), 1e-300))
    s = stats(d_on)
    s["frac_windows_suppressed"] = float((d_on > 0).mean())
    s["per_window_delta"] = d_on.tolist()
    s["joint_delta_log10_mean"] = float(d_j.mean())
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                             (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lg_on = logit(sham_cell["P"].mean(axis=0)) - logit(read_cell["P"].mean(axis=0))
    s["logit_delta"] = stats(lg_on)
    Ps, Pr = sham_cell["P"], read_cell["P"]
    Bs, Br, n = Ps.shape[0], Pr.shape[0], Ps.shape[1]
    brng = np.random.default_rng(seed)
    hs, hr = max(1, Bs // 2), max(1, Br // 2)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        bs_ = brng.integers(0, Bs, hs)
        br_ = brng.integers(0, Br, hr)
        w_ = brng.integers(0, n, n)
        draws[i] = Ps[np.ix_(bs_, w_)].mean() - Pr[np.ix_(br_, w_)].mean()
    s["delta_bootstrap_ci95"] = [float(np.percentile(draws, 2.5)),
                                 float(np.percentile(draws, 97.5))]
    return s


# ------------------------------------------------------------------ main
def main():
    rd = run_dir("e081")
    log("E081 RIF probe: does reading one name suppress its coordinate "
        "neighbor? (P-A; eval-only, no weight edits)")

    corpus, train_text, install_occ, probes, mix, held_mix = rebuild_protocol()
    pools, pool_flags = build_pools(train_text, install_occ, probes)
    log(f"protocol rebuilt: install60 {mix}, held30 {held_mix}; probes "
        f"FL={len(probes['FL'])} EL={len(probes['EL'])} Z={len(probes['Z'])}; "
        f"pools "
        f"{ {f'r{r}p{p}': len(v) for (r, p), v in pools.items()} }")

    CELLS = [  # (read, probe, n_res) — order fixes RNG seeds
        ("FL", "EL", B_RESAMPLE), ("sham", "EL", B_RESAMPLE),         # primary
        ("EL", "FL", B_RESAMPLE), ("sham", "FL", B_RESAMPLE),
        ("FL", "FL", B_SECONDARY), ("EL", "EL", B_SECONDARY),         # self
        ("Z", "Z", B_SECONDARY), ("sham", "Z", B_SECONDARY),          # Z legs
        ("FL", "Z", B_SECONDARY), ("EL", "Z", B_SECONDARY),
        ("Z", "FL", B_SECONDARY), ("Z", "EL", B_SECONDARY),
        ("scrEL", "FL", B_SECONDARY), ("scrFL", "EL", B_SECONDARY),   # dev 8
    ]
    results, gates = {}, {}
    for ck_name, ck_file in CKPTS.items():
        net = load(E43.REPO / "runs" / "checkpoints" / ck_file)
        w_sig = (net.wpe.weight.data.clone(), net.wte.weight.data.clone())
        log(f"net {ck_name}: params {net.num_params():,}")
        res = {}
        # canonical baselines first (instrument gates)
        for probe_key in ("FL", "EL", "Z"):
            rng = _random.Random(8100 + 900 + {"FL": 3, "EL": 1, "Z": 7}[probe_key])
            c = run_cell(net, corpus, train_text, "sham", probe_key,
                         probes[probe_key], None, rng, shifted=False)
            res[f"base__probe{probe_key}"] = c
            a = arm_stats(c)
            log(f"[{ck_name}] baseline probe-{probe_key} @129: "
                f"p(first) {a['onset']['mean']:.4f} "
                f"[{a['onset']['ci95'][0]:.4f},{a['onset']['ci95'][1]:.4f}] "
                f"joint {a['joint_mean']:.2e}")
        gz = res["base__probeZ"]["P"].mean()
        drift = abs(gz - Z_GATE[ck_name])
        gates[ck_name] = {"pz_canonical": float(gz),
                          "pz_published": Z_GATE[ck_name],
                          "drift": float(drift),
                          "pass": bool(drift < 5e-3)}
        assert gates[ck_name]["pass"], (f"{ck_name} Z gate drift {drift:.2e} "
                                        f"(protocol identity failed)")
        log(f"[{ck_name}] instrument gate: p(Z)@129 {gz:.7f} vs published "
            f"{Z_GATE[ck_name]:.7f} (|drift| {drift:.2e}, OK)")
        # shifted arms
        for ci, (read, probe, n_res) in enumerate(CELLS):
            rng = _random.Random(8100 + ci * 100 + b_seed(ck_name))
            pool_key = read[3:] if read.startswith("scr") else read
            c = run_cell(net, corpus, train_text, read, probe,
                         probes[probe], pools.get((pool_key, probe)), rng,
                         shifted=True, n_res=n_res)
            res[f"read{read}__probe{probe}"] = c
            a = arm_stats(c)
            log(f"[{ck_name}] read-{read:5s} -> probe-{probe} @243: "
                f"p(first) {a['onset']['mean']:.4f} "
                f"[{a['onset']['ci95'][0]:.4f},{a['onset']['ci95'][1]:.4f}] "
                f"joint {a['joint_mean']:.2e} ({a['n_sessions']} sessions)")
        # no-weight-edit check
        assert torch.equal(net.wpe.weight.data, w_sig[0]) and \
            torch.equal(net.wte.weight.data, w_sig[1]), "weights changed!"
        log(f"[{ck_name}] no-weight-edit check passed")
        results[ck_name] = res

    # ---- deltas, gates, verdict
    verdicts, deltas_all = {}, {}
    for ck_name, res in results.items():
        d = {}
        # primary cross directions
        d["cross_FL_to_EL"] = delta_cell(res["readsham__probeEL"],
                                         res["readFL__probeEL"])
        d["cross_EL_to_FL"] = delta_cell(res["readsham__probeFL"],
                                         res["readEL__probeFL"])
        # secondary: self + Z + scramble legs (same statistic, non-gating)
        for tag, sham_k, read_k in (
                ("self_FL", "readsham__probeFL", "readFL__probeFL"),
                ("self_EL", "readsham__probeEL", "readEL__probeEL"),
                ("zread_to_Z", "readsham__probeZ", "readZ__probeZ"),
                ("FLread_to_Z", "readsham__probeZ", "readFL__probeZ"),
                ("ELread_to_Z", "readsham__probeZ", "readEL__probeZ"),
                ("Zread_to_FL", "readsham__probeFL", "readZ__probeFL"),
                ("Zread_to_EL", "readsham__probeEL", "readZ__probeEL"),
                ("scr_EL_to_FL", "readsham__probeFL", "readscrEL__probeFL"),
                ("scr_FL_to_EL", "readsham__probeEL", "readscrFL__probeEL")):
            d[tag] = delta_cell(res[sham_k], res[read_k])
        # sham noise gate: placebo split of sham resamples (halves)
        gate = {}
        for probe_key in ("FL", "EL", "Z"):
            Ps = res[f"readsham__probe{probe_key}"]["P"]
            h = Ps.shape[0] // 2
            placebo = (Ps[:h].mean(axis=0) - Ps[h:].mean(axis=0)).mean()
            gate[probe_key] = float(placebo)
        gate_pass = all(abs(gate[k]) < SHAM_NOISE_BAR for k in ("FL", "EL"))
        # shift cost (honesty number, not gated)
        shift = {pk: float(res[f"readsham__probe{pk}"]["P"].mean()
                           - res[f"base__probe{pk}"]["P"].mean())
                 for pk in ("FL", "EL", "Z")}
        fired = []
        for direction in ("cross_FL_to_EL", "cross_EL_to_FL"):
            s = d[direction]
            if s["mean"] >= SUPP_BAR and s["ci95"][0] > 0:
                fired.append(direction)
        rif = bool(fired) and gate_pass
        # scramble-control reading (deviation 8, non-gating): does the firing
        # direction's effect survive the anagram?
        scr = {"cross_FL_to_EL": d["scr_FL_to_EL"],
               "cross_EL_to_FL": d["scr_EL_to_FL"]}
        scr_note = {k: {"mean": sc["mean"], "ci95": sc["ci95"],
                        "boot_ci95": sc["delta_bootstrap_ci95"]}
                    for k, sc in scr.items()}
        if rif:
            v = (f"RIF PRESENT on {ck_name} — cross-name suppression "
                 f">= {SUPP_BAR} absolute with CI excluding 0 and sham "
                 f"noise gate passed; fired directions: {fired}")
        elif fired:
            v = (f"PARTIAL on {ck_name} — suppression bar reached "
                 f"({fired}) but sham noise gate FAILED {gate} "
                 f"(deviation-2 caveat; report honestly)")
        else:
            v = (f"NULL on {ck_name} — no cross-name reading effect >= "
                 f"{SUPP_BAR} absolute: reads are pure at this scale "
                 f"(the P-A answer); largest cross delta "
                 f"{max(d['cross_FL_to_EL']['mean'], d['cross_EL_to_FL']['mean']):+.4f}")
        verdicts[ck_name] = {"fired": fired, "gate_placebo": gate,
                             "gate_pass": bool(gate_pass),
                             "shift_cost_sham_minus_base": shift,
                             "scramble_control_report_only": scr_note,
                             "rif": rif, "verdict": v}
        deltas_all[ck_name] = d
        log(f"[{ck_name}] supp FL->EL {d['cross_FL_to_EL']['mean']:+.4f} "
            f"CI {d['cross_FL_to_EL']['ci95']} boot "
            f"{[round(x, 4) for x in d['cross_FL_to_EL']['delta_bootstrap_ci95']]} "
            f"| supp EL->FL {d['cross_EL_to_FL']['mean']:+.4f} CI "
            f"{d['cross_EL_to_FL']['ci95']} boot "
            f"{[round(x, 4) for x in d['cross_EL_to_FL']['delta_bootstrap_ci95']]} "
            f"| placebo gate {gate} (pass {gate_pass}) | shift cost {shift}")
        log(f"[{ck_name}] scramble controls: scrFL->EL "
            f"{d['scr_FL_to_EL']['mean']:+.4f} (boot "
            f"{[round(x, 4) for x in d['scr_FL_to_EL']['delta_bootstrap_ci95']]}) "
            f"| scrEL->FL {d['scr_EL_to_FL']['mean']:+.4f} (boot "
            f"{[round(x, 4) for x in d['scr_EL_to_FL']['delta_bootstrap_ci95']]})")
        log(f"[{ck_name}] VERDICT: {v}")

    if "e048_repro" in verdicts:
        agree = (verdicts["e048_dose"]["rif"] == verdicts["e048_repro"]["rif"])
        note = ("; e048_repro agrees (n=2)" if agree
                else "; e048_repro DISAGREES — honest mixed")
    else:
        note = " (smoke: dose only)"
    if verdicts["e048_dose"]["rif"]:
        overall = "RIF PRESENT on the primary net (e048_dose)" + note
    else:
        overall = ("NULL on the primary net (e048_dose) — reads are pure "
                   "(the P-A answer)" + note)
    log(f"OVERALL: {overall}")

    # ---- metrics
    def cells_json(res):
        return {k: {**arm_stats(c), "row": c["row"], "sess_len": c["sess_len"],
                    "P_flat": c["P"].flatten().tolist()}
                for k, c in res.items()}

    out = {
        "experiment": "e081_rif_probe",
        "proposal": "P-A (scratch/explorations_harvest_20260926.md)",
        "registered": REGISTERED,
        "protocol": {
            "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
            "install_mix": mix, "held_mix": held_mix,
            "probe_battery_sizes": {k: len(v) for k, v in probes.items()},
            "prefix_len": PREFIX, "separator": SEP,
            "readout_rows": {"shifted": ROW_SHIFT, "canonical": ROW_CANON},
            "elicitation_pool_sizes": {f"read{r}__probe{p}": len(v)
                                       for (r, p), v in pools.items()},
            "pool_flags": pool_flags,
            "resamples": {"primary": B_RESAMPLE, "secondary": B_SECONDARY,
                          "bootstrap_draws": N_BOOT},
            "name_read_mass_tokens": {"FL": 16, "EL": 18, "Z": 14},
        },
        "gates_instrument": gates,
        "cells": {ck: cells_json(res) for ck, res in results.items()},
        "deltas_suppression": {
            ck: {k: {kk: vv for kk, vv in v.items() if kk != "per_window_delta"}
                 for k, v in d.items()} for ck, d in deltas_all.items()},
        "verdicts": verdicts,
        "overall": overall,
        "bars": {"supp_bar": SUPP_BAR, "sham_noise_bar": SHAM_NOISE_BAR,
                 "ci_z": CI_Z},
        "elapsed_s": round(time.time() - T0, 1),
        "cpu_threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", out)

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
    ck = "e048_dose"
    ck_list = [c for c in ("e048_dose", "e048_repro") if c in results]
    res, d, verd = results[ck], deltas_all[ck], verdicts[ck]

    # (0,0) primary: onset p per probe under read-other / sham / canonical base
    ax = axes[0, 0]
    xs = np.arange(2)
    bw = 0.13
    series = [
        ("read-OTHER name", ["readFL__probeEL", "readEL__probeFL"], "crimson"),
        ("sham (matched pos)", ["readsham__probeEL", "readsham__probeFL"], "steelblue"),
        ("read-SELF name", ["readEL__probeEL", "readFL__probeFL"], "darkorange"),
        ("canonical base @129", ["base__probeEL", "base__probeFL"], "gray"),
    ]
    for j, (lbl, keys, col) in enumerate(series):
        for rep, ckn in enumerate(ck_list):
            vals = [arm_stats(results[ckn][k])["onset"]["mean"] for k in keys]
            errs = [arm_stats(results[ckn][k])["onset"]["sem"] * CI_Z
                    for k in keys]
            ax.bar(xs + (j * 2 + rep - 3.5) * bw, vals, bw * 0.92, yerr=errs,
                   capsize=2, color=col, alpha=1.0 if ckn == ck else 0.35,
                   edgecolor="k", linewidth=0.4,
                   label=lbl + (" (dose)" if ckn == ck and j in (0, 1)
                                else (" (repro)" if ckn != ck and j in (0, 1)
                                      else "")))
            for x, v in zip(xs + (j * 2 + rep - 3.5) * bw, vals):
                ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.2,
                        rotation=90, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(["probe ELIZABETH\n(61 windows)", "probe FLORIZEL\n(29 windows)"])
    ax.set_ylabel("onset p(first name char)")
    s1, s2 = d["cross_FL_to_EL"], d["cross_EL_to_FL"]
    ax.set_title(f"PRIMARY (e048_dose): reading the OTHER name vs sham, "
                 f"matched row 243\nsupp FL->EL {s1['mean']:+.4f} "
                 f"CI[{s1['ci95'][0]:+.4f},{s1['ci95'][1]:+.4f}] | "
                 f"supp EL->FL {s2['mean']:+.4f} "
                 f"CI[{s2['ci95'][0]:+.4f},{s2['ci95'][1]:+.4f}]", fontsize=9)
    ax.legend(fontsize=6.5, ncols=2)
    ax.set_ylim(0, 1.02)

    # (0,1) per-window suppression distributions (primary)
    ax = axes[0, 1]
    for i, (key, lbl, col) in enumerate((("cross_FL_to_EL",
                                          "read FLORIZEL -> probe ELIZABETH",
                                          "crimson"),
                                         ("cross_EL_to_FL",
                                          "read ELIZABETH -> probe FLORIZEL",
                                          "steelblue"))):
        dw = np.array(d[key]["per_window_delta"])
        ax.hist(dw, bins=12, alpha=0.55, color=col, label=
                f"{lbl}: mean {dw.mean():+.4f}, "
                f"{int((dw > 0).sum())}/{len(dw)} windows supp",
                edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(SUPP_BAR, color="seagreen", ls="--", lw=1.4,
               label=f"RIF bar +{SUPP_BAR}")
    ax.axvline(-SUPP_BAR, color="purple", ls=":", lw=1.2,
               label=f"facilitation -{SUPP_BAR}")
    ax.set_xlabel("per-window delta: p(sham) - p(read)  [positive = read suppresses]")
    ax.set_ylabel("probe windows")
    ax.set_title("PRIMARY: per-window cross-name suppression distributions "
                 "(e048_dose)", fontsize=9.5)
    ax.legend(fontsize=7.5)

    # (1,0) secondary legs: suppression (sham - read) per leg, both nets
    ax = axes[1, 0]
    legs = ["cross_FL_to_EL", "cross_EL_to_FL", "self_FL", "self_EL",
            "zread_to_Z", "FLread_to_Z", "ELread_to_Z", "Zread_to_FL",
            "Zread_to_EL", "scr_FL_to_EL", "scr_EL_to_FL"]
    lbls = ["FL->EL\n(CROSS a)", "EL->FL\n(CROSS c)", "FL->FL\n(self)",
            "EL->EL\n(self)", "Z->Z\n(read install)", "FL->Z", "EL->Z",
            "Z->FL", "Z->EL", "scrFL->EL\n(anagram)", "scrEL->FL\n(anagram)"]
    xs = np.arange(len(legs))
    for rep, (ckn, col) in enumerate(zip(ck_list, ("crimson", "steelblue"))):
        vals = [deltas_all[ckn][k]["mean"] for k in legs]
        errs = [CI_Z * (v["sem"] if "sem" in v else 0)
                for v in [deltas_all[ckn][k] for k in legs]]
        ax.bar(xs + (rep - 0.5) * 0.38, vals, 0.36, yerr=errs, capsize=2,
               color=col, alpha=1.0 if ckn == ck else 0.4, edgecolor="k",
               linewidth=0.4, label=ckn)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(SUPP_BAR, color="seagreen", ls="--", lw=1.2,
               label=f"RIF bar +{SUPP_BAR}")
    ax.axhline(-SUPP_BAR, color="purple", ls=":", lw=1.2, label="facilitation bar")
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls, fontsize=7)
    ax.set_ylabel("delta: p(sham) - p(read)")
    ax.set_title("SECONDARY legs (non-gating): self-priming and installed-fact "
                 "(Z) legs — positive = the read SUPPRESSES the probe",
                 fontsize=9.5)
    ax.legend(fontsize=8)

    # (1,1) resample variability + shift cost honesty panel
    ax = axes[1, 1]
    for i, (key, col) in enumerate((("readFL__probeEL", "crimson"),
                                    ("readsham__probeEL", "steelblue"),
                                    ("readEL__probeFL", "darkorange"),
                                    ("readsham__probeFL", "seagreen"))):
        rm = res[key]["P"].mean(axis=1)
        ax.plot(np.full(B_RESAMPLE, i) + np.random.uniform(-0.13, 0.13,
                                                           B_RESAMPLE),
                rm, "o", ms=3, color=col, alpha=0.6)
        ax.hlines(rm.mean(), i - 0.22, i + 0.22, color="k", lw=1.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["read FL\nprobe EL", "sham\nprobe EL",
                        "read EL\nprobe FL", "sham\nprobe FL"], fontsize=8)
    ax.set_ylabel("per-resample mean onset p (elicitation-set resamples)")
    sc = verd["shift_cost_sham_minus_base"]
    gate = verd["gate_placebo"]
    ax.set_title(f"HONESTY: resample spread + gates (e048_dose)\n"
                 f"shift cost sham@243-base@129: "
                 f"FL {sc['FL']:+.3f} EL {sc['EL']:+.3f} Z {sc['Z']:+.3f} "
                 f"(NOT gated, deviation 2) | placebo gate "
                 f"{ {k: round(v, 5) for k, v in gate.items()} } "
                 f"(pass {verd['gate_pass']})", fontsize=8)

    fig.suptitle(f"e081 RIF probe (P-A) — OVERALL: {overall}\n"
                 f"reads-are-pure test at the shared name-onset slot; "
                 f"no training, no weight edits; B={B_RESAMPLE} resamples",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rd / "rif_probe.png", dpi=140)
    log(f"done -> {rd}")
    return 0


def b_seed(ck_name):
    return 0 if ck_name == "e048_dose" else 5000


if __name__ == "__main__":
    sys.exit(main())
