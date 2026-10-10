#!/usr/bin/env python
"""
x32 — THE FIRST-STEP ATLAS (the formation frontier's CPU cell; x34's
       follow-through). One step per committed install draw, recomputed
       DETERMINISTICALLY on CPU, fingerprinted geometrically. PURE CPU by
       construction (torch.set_num_threads(4), map_location='cpu', no CUDA
       call anywhere). No NOTES/THINKING/QUEUE/STATE writes (the heartbeat
       folds this cell).

=============================================================================
THE QUESTION (frozen VERBATIM from the dispatch, 2026-10-10)
=============================================================================
WHAT MAKES THE SAME RIG SURGE (>= 1e4x prior at step one, instant bulky
formation) ON ONE NAME-CONTEXT PAIR AND NOT ANOTHER (no-surge ~1.0x, slow
pruned formation)? Every lab-installed fresh name took the no-surge branch;
the surge belongs to canonical-fact reinstalls. x34's ratio axis re-worded
the fork NO-SURGE vs SURGE; this atlas asks whether the fork is visible in
the FIRST STEP's GEOMETRY at all.

=============================================================================
THE CORPUS (the dispatch's named set; every gen read from committed
artifacts at runtime — no retyped numbers)
=============================================================================
The 16 draws x34 inventoried with recoverable first steps, ALL run through
e261's chunked_install family (G_PROTOIDENT in every parent):
  - THE CANON: gen 24314 (ZEPHYRA, root g1c, K10K room) — the no-surge
    exemplar (s1/prior 1.006); its committed s1 triple-bound by x34.
  - e324's four fresh draws (ZEPHYRA, root, K10K held) — surge.
  - e327's eight spread draws ((2i+1)*2^28) — surge.
  - THE PRIOR-FLAT SET: TAVIREN 31102, QELVARO 32801, NYSTORA 33002
    (parasite names at the HOST's contexts, base e001, K10K room) —
    no-surge (ratios 1.002).
RIDER (disclosed, excluded from adjudication): the canon's first batch with
NO room projection (mode FREE — e264's FREE arm's committed s1 exists for
the fidelity cross-check). It isolates the room's contribution to the
canon's step at held batch.

The step itself is NOT in any committed artifact (x34's G_PROTOID_AGG
disclosure: the step-one parameter-delta decomposition is recorded nowhere)
— so this cell RECOMPUTES step one per draw: the e261 step body VERBATIM
(same banks, same batch arithmetic, same clip, same room hook, same AdamW,
same house cosine so step-1 lr = 1e-3 x cosine_lr(0,1000) = 1e-5), on CPU.
The committed s1 values (read from the parents, md5-bound) serve as the
fidelity check, not the datum: |log10(s1_cpu/s1_committed)| <= 1 decade AND
same side of the fork (surge/no-surge by x34's ratio axis) per draw. The
GEOMETRY is computed on ONE platform for all draws (internal consistency);
platform drift (committed GPU vs this CPU) is disclosed here once and
validated by the fidelity gate.

=============================================================================
THE FINGERPRINT (frozen axes per draw)
=============================================================================
Given theta0 = the draw's pre-install state (root for ZEPHYRA draws; the
settled e001 base for the parasite draws — the parents' own convention),
r = grad_theta [mean_60 log p(name_char | g0 battery ctx)] at theta0 (the
NAME'S READ-GRADIENT DIRECTION), g1 = the first APPLIED gradient (post-clip,
post-room-projection — what the hook writes back), d1 = theta1 - theta0
(the EFFECTIVE first step: after opt.step AND the eval path's wall settle,
i.e. the displacement the committed s1 read actually saw):
  F1 ||d1||            — the first step's parameter-delta norm (raw and
                         wall-corrected co-reported)
  F2 in-room fraction  — ||P_K10K(d1)||/||d1|| (the room projector's
                         decomposition; the dispatch's "victim's room basis")
  F3 cos(d1, -sign(g1))— the SIGN-RAY ALIGNMENT (e231's opt-era law: AdamW's
                         first displacement IS minus the sign-ray; expected
                         ~1 BY RECIPE — recipe-pinned, cannot be a hammer)
  F4 dose_read(d1)     — d1 . r/||r|| (the step's DOSE onto the name's
                         read-gradient; the first-order predicted log-read
                         move, in nats — the dispatch's named statistic)
  F5 cos(d1, r)        — the direction form of the dose
  F6 dose_read(g1)     — the applied gradient's dose (pre-Adam geometry)
  F7 cos(g1, r)        — the applied gradient's direction form
  + co-columns: kept_frac = ||g1||/||g_clip||, ||r||, prior_cpu (the read at
    theta0), s1_cpu, wall_correction = ||theta1_raw - theta1_eff||.

=============================================================================
THE BARS (frozen VERBATIM from the dispatch, BEFORE any compute)
=============================================================================
  - ONE-HAMMER: "a single geometric statistic — e.g., the first step's
    projection onto the name-read-gradient — separates surge from no-surge
    across ALL draws at zero overlap" — the fork IS first-step-geometric.
  - THREE-DEATHS: "no separation; the textures are not first-step-geometric"
    — no common geometric signature.

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses, they
do not move the bars):
  * Classes: SURGE := x34 side SURVIVE (ratio >= ~1e4); NO-SURGE := x34
    sides DIE + PRIOR-FLAT? (ratio ~ 1.0) — labels READ from runs/x34/
    metrics.json the_census.records (md5-bound), never retyped.
  * Candidate statistics := F1..F7 above, EXACTLY these seven, adjudicated
    as a frozen list — no eighth statistic joins after compute.
  * "separates at zero overlap" := max(class-A) < min(class-B) or
    max(class-B) < min(class-A) over the 16 adjudication draws (the rider
    excluded). Co-reported per statistic: the ZEPHYRA-matched subset
    (13 draws: the canon + 12 surge, same name/battery/room — the parasite
    rows are cross-name and disclosed as such).
  * Verdict precedence: ONE-HAMMER (>= 1 of the 7 separates all 16 at zero
    overlap; the hammer(s) named) > THREE-DEATHS.
  * F3 is recipe-pinned (e231: the first displacement IS -lr(sign(g)+wd*theta)
    to ~1e-4 by construction) — if F3 "separates" it is reported with the
    recipe-identity disclosure and the verdict carries the caveat.

=============================================================================
P-x32a (registered BEFORE compute)
=============================================================================
  Lab lean (dispatch verbatim): "the lab leans NONE (the honest lean after
  twelve; state your own)."
  Executor's own read (registered at birth): ONE-HAMMER, moderately, via F4
  (dose_read). Grounds: (i) all 12 surge draws land at the SAME s1 (0.7449-
  0.7453, spread 0.05%) — a uniform ceiling, while the canon stays at prior
  in FOUR rooms INCLUDING unprojected FREE (e264's arms) — the batch, not
  the room, carries the fork; (ii) x34's floor: NO draw ever read below its
  prior at step one (min ratio 1.0019) — the no-surge steps are ~ORTHOGONAL
  to the read direction (dose ~ 0), never opposite; a cliff-crossing surge
  needs dose > 0. Countervailing (registered): the first-order budget is
  ||d1|| ~ 1e-3 nats of log-read move — CANNOT kinetically explain a ~10-
  nat surge (1.3e-5 -> 0.745); if the crossing is a nonlinear threshold
  whose normal is NOT the read-gradient, the dose need not separate, and
  the hammer dies into THREE-DEATHS. The hammer claim is CORRELATIONAL
  (geometry predicts texture), never kinetic.

=============================================================================
P-e328x32 (the registration the dispatch orders BEFORE e328's compute)
=============================================================================
  The MAPPING RULE (frozen at birth, so the call is data-derived but never
  shopped): the dispatch's own conditional — "fork-in-the-batch predicts
  the swap flips the canon to surge IF the first batch's content carries
  the surge geometry; fork-in-the-gen predicts no flip." Frozen mapping:
    atlas verdict ONE-HAMMER  -> P-e328x32 = FLIPS-TO-SURGE (the first
                                 batch carries a geometry that separates
                                 the textures; e328's swap arm predicts
                                 s1/prior >= 1e3)
    atlas verdict THREE-DEATHS-> P-e328x32 = STAYS-NO-SURGE (the first
                                 batch carries NO distinguishing geometry;
                                 e328's swap arm predicts s1/prior ~ 1)
  The CONCRETE CALL (with the atlas's numbers) is written by the COMPLETE
  run into metrics['P_e328x32'] + the REPORT and committed BEFORE e328's
  birth. This cell never runs e328's compute.

=============================================================================
PROTOCOL (house rules honored)
=============================================================================
  ONE file (this). BIRTH COMMIT first (this docstring = the frozen
  registration; pushed BEFORE compute), then smoke (runs/x32_smoke/,
  gitignored), then the full run. Outputs runs/x32/{metrics.json,
  REPORT.md, x32_first_step_atlas.png}. Builds on: e231 (the sign-ray
  recipe law + its instrument autopsy), e261/e264 (the install rig + the
  canon), e272 (the room wheel), e290 (the canon write norm), e291/e311/
  e321/x28/e330 (the install families + the parasite installs), e323/e324/
  e327 (the fork census), x34 (the s1 texture census + the corpus). What
  is NEW: the first step's own geometry — norm, room decomposition,
  sign-ray, and the read-direction dose — across the whole recoverable
  corpus on one platform; the separation adjudication; the registered
  branch prediction for e328's batch swap.

Run:  python lab/x32_first_step_atlas.py            (full)
      X32_SMOKE=1 python lab/x32_first_step_atlas.py (smoke, 3 draws)

AMENDMENTS (dated, post-birth repairs — disclosed, no bar ever moved):
- 2026-10-10 SMOKE CATCH 1 (the canon "flip") + SMOKE CATCH 2 (THE NET0
  CONFOUND — the cell's real finding). Catch 1: the canon row first ran
  from the g1c ROOT (my error) and "flipped" (CPU s1 0.745 vs committed
  1.3466e-05) while the instrument validated exactly on e324/32401 and
  TAVIREN. Chasing it produced Catch 2: the committed canon's step-1
  ledger fingerprint (ce_batch 1.0701713562011719) does NOT match a
  root-start step — but matches BIT-EXACTLY a step from THE BASE e001.
  Source check: e264's run_arm and e272's K10KR both pass
  G1.evl_load(base_sd) as net0 (THE canon's actual convention), while
  e324/e327's draws pass the g1c ROOT. From the base, this cell
  reproduces the canon's committed record EXACTLY (ce and s1 identical
  to all 16 digits). THE NET0 CONFOUND: the "SURGE" class (e324 x4 +
  e327 x8) started from the FORMED root (standing g0 read 0.7447534) —
  their s1 ~ 0.745 is the STANDING READ STAYING (ratio-to-standing
  1.0002), which reads as ~5.5e4x on x34's e001-prior axis; the
  "NO-SURGE" class (canon + parasites) started from the base — s1 =
  prior x 1.002-1.006. AT STEP ONE NOTHING EVER HAPPENS ON EITHER
  SIDE: every one of the 16 draws' first step is a no-event on its own
  state's prior. THE S1 FORK OF x34/e324/e327 IS (almost entirely) A
  STARTING-STATE CONFOUND, not a first-step event. REPAIRS at smoke,
  none touching a bar or the candidate list: (a) net0 per lineage
  corrected to each parent's ACTUAL convention (canon: base; e324/e327:
  root; parasites: base) — a harness repair, disclosed; (b) the
  fidelity gate restored to its strict birth form (all 16 draws must
  reproduce); (c) the knife-edge probe reframed as the NO-EVENT probes
  (t-scan + sign-flip, now on the correct states); (d) P6c re-written
  as THE NET0 CONFOUND section; (e) P-e328x32's countervailing carries
  the confound (the call logic untouched — frozen at birth). The class
  boundary of the frozen adjudication (SURVIVE vs DIE) coincides with
  the net0 boundary — any zero-overlap "hammer" would separate STARTING
  STATES, not step-one textures; the verdict interpretation carries
  this verbatim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402
import torch.nn.functional as F                              # noqa: E402

import common                                                # noqa: E402
from common import CharCorpus, cosine_lr                     # noqa: E402

import e043_install as E43                                   # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG)
import g1b_continuity as GB                                  # noqa: E402 — MUST
                                                      # be imported BEFORE G1
import g1_anchored_ball as G1                                # noqa: E402
import e261_rank_ladder as E261                              # noqa: E402 — the
                                                      # committed rig module

torch.set_num_threads(4)          # the desk cap; CPU-only by construction

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

SMOKE = bool(os.environ.get("X32_SMOKE"))
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / ("x32_smoke" if SMOKE else "x32")
OUT.mkdir(parents=True, exist_ok=True)
CPU = torch.device("cpu")

T0 = datetime.now(timezone.utc)
METRICS: dict = {
    "experiment": "x32_first_step_atlas" + ("_smoke" if SMOKE else ""),
    "phase": "BIRTH-RUNNING",
    "date": T0.isoformat(),
    "smoke": SMOKE,
    "cpu_only_by_construction": True,
    "torch_threads": 4,
}


def log(msg: str) -> None:
    print(f"[x32] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(rel: str) -> dict:
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def get(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError("/".join(map(str, path)))
        cur = cur[p]
    return cur


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


# =============================================================================
# P0 — the artifact register (md5-bind EVERY artifact this cell reads)
# =============================================================================
ARTIFACTS = [
    "runs/e264/metrics.json",          # the canon's committed s1 (K10K+FREE)
    "runs/e324/metrics.json",          # the four fresh draws' committed s1
    "runs/e327/metrics.json",          # the eight spread draws' committed s1
    "runs/e330/metrics.json",          # NYSTORA's write (s1 + gen)
    "runs/e311/metrics.json",          # TAVIREN install context (G_FRESH)
    "runs/x28/metrics.json",           # QELVARO install context (G_QELINST)
    "runs/x34/metrics.json",           # THE corpus + labels (the taxonomy)
    "runs/x24/metrics.json",           # the priors (cross-check only)
    "runs/checkpoints/e311_TAVINST_resume.pt",   # TAVIREN's committed s1
    "runs/checkpoints/x28_QELINST_resume.pt",    # QELVARO's committed s1
    "runs/checkpoints/g1c_root.pt",    # the ZEPHYRA draws' pre-install state
    "runs/checkpoints/e001.pt",        # the parasite draws' pre-install state
    "runs/checkpoints/e246_late_span.pt",        # the span (LadderRooms)
    "runs/checkpoints/e258_vmap.pt",             # the v-map (LadderRooms)
    "runs/checkpoints/e264_rooms.pt",            # the held room's bit-bind
]
G_PARENTS: dict = {"form": "every artifact this atlas reads, md5-bound at "
                           "runtime (files never modified; values read at "
                           "runtime, never retyped)",
                   "md5": {}, "pass": True}
for rel in ARTIFACTS:
    p = ROOT / rel
    if not p.exists():
        G_PARENTS["md5"][rel] = "MISSING"
        G_PARENTS["pass"] = False
    else:
        G_PARENTS["md5"][rel] = md5_of(p)
METRICS["gates"] = {"G_PARENTS": G_PARENTS}
log(f"P0: {len(ARTIFACTS)} artifacts md5-bound "
    f"({'PASS' if G_PARENTS['pass'] else 'MISSING FILES'})")

# =============================================================================
# P1 — the corpus + labels (READ from x34's committed census; never retyped)
# =============================================================================
x34 = jload("runs/x34/metrics.json")
e324 = jload("runs/e324/metrics.json")
e327 = jload("runs/e327/metrics.json")
e330 = jload("runs/e330/metrics.json")
e264 = jload("runs/e264/metrics.json")

x34_records = {f"{r['lineage']}|{r['gen']}": r
               for r in get(x34, "the_census", "records")}


def corpus_rows():
    """The draw list: gens + committed s1 + labels, all READ from the
    committed artifacts at runtime (x34's records + the parents' own rows,
    cross-checked against each other)."""
    rows = []
    canon_gen = int(get(e324, "the_census", "canon", "gen"))
    canon_s1 = get(e264, "arms", "K10K", "install", "traj")[0]["g0_pz"]
    rec = x34_records[f"canon|{canon_gen}"]
    assert abs(rec["s1_g0"] - float(canon_s1)) <= 1e-12, "canon s1 drift"
    rows.append({
        "draw": "canon", "gen": canon_gen, "name": "ZEPHYRA",
        "side": rec["side"], "s1_committed": float(canon_s1),
        "prior_committed": float(rec["prior"]),
        "s1_source": "runs/e264/metrics.json arms.K10K.install.traj[0]",
    })
    for g in get(e324, "the_fresh_gens", "gens"):
        s1 = get(e324, "draws", f"GEN{g}", "s1_g0")
        rec = x34_records[f"e324|{g}"]
        assert abs(rec["s1_g0"] - s1) <= 1e-12, f"e324 GEN{g} s1 drift"
        rows.append({
            "draw": f"e324/{g}", "gen": int(g), "name": "ZEPHYRA",
            "side": rec["side"], "s1_committed": float(s1),
            "prior_committed": float(rec["prior"]),
            "s1_source": f"runs/e324/metrics.json draws.GEN{g}.s1_g0",
        })
    for g in get(e327, "the_fresh_gens", "gens"):
        rec = next(r for r in get(e327, "the_census", "census_rows")
                   if r["gen"] == g and r["label"] != "CANON")
        x34r = x34_records[f"e327|{g}"]
        assert abs(x34r["s1_g0"] - rec["s1_g0"]) <= 1e-12, \
            f"e327 GEN{g} s1 drift"
        rows.append({
            "draw": f"e327/{g}", "gen": int(g), "name": "ZEPHYRA",
            "side": rec["side"], "s1_committed": float(rec["s1_g0"]),
            "prior_committed": float(x34r["prior"]),
            "s1_source": f"runs/e327/metrics.json the_census.census_rows "
                         f"GEN{g}",
        })
    tav_tr = torch.load(ROOT / "runs/checkpoints/e311_TAVINST_resume.pt",
                        map_location="cpu", weights_only=False)["traj"]
    qel_tr = torch.load(ROOT / "runs/checkpoints/x28_QELINST_resume.pt",
                        map_location="cpu", weights_only=False)["traj"]
    nys_s1 = get(e330, "the_write", "s1")
    for lineage, name, s1, src in (
            ("e311", "TAVIREN", tav_tr[0]["g0_pz"],
             "runs/checkpoints/e311_TAVINST_resume.pt traj[0]"),
            ("x28", "QELVARO", qel_tr[0]["g0_pz"],
             "runs/checkpoints/x28_QELINST_resume.pt traj[0]"),
            ("e330", "NYSTORA", nys_s1,
             "runs/e330/metrics.json the_write.s1")):
        rec = next(r for r in x34_records.values()
                   if r["lineage"] == lineage)
        assert abs(rec["s1_g0"] - float(s1)) <= 1e-12, f"{name} s1 drift"
        rows.append({
            "draw": f"{lineage}/{name}", "gen": int(rec["gen"]), "name": name,
            "side": rec["side"], "s1_committed": float(s1),
            "prior_committed": float(rec["prior"]), "s1_source": src,
        })
    return rows


ROWS = corpus_rows()
CANON_FREE_S1 = get(e264, "arms", "FREE", "install", "traj")[0]["g0_pz"]
assert len(ROWS) == 16, f"corpus size {len(ROWS)} != 16"
SURGE = [r for r in ROWS if r["side"] == "SURVIVE"]
NOSURGE = [r for r in ROWS if r["side"] in ("DIE", "PRIOR-FLAT?")]
assert len(SURGE) == 12 and len(NOSURGE) == 4, \
    f"class sizes {len(SURGE)}/{len(NOSURGE)} != 12/4"
if SMOKE:
    keep = {"canon", "e324/32401", "e311/TAVIREN"}
    ROWS_ADJ = [r for r in ROWS if r["draw"] in keep]
else:
    ROWS_ADJ = ROWS
log(f"P1: corpus {len(ROWS)} draws (surge {len(SURGE)} / no-surge "
    f"{len(NOSURGE)}); adjudicating {len(ROWS_ADJ)} at "
    f"{'smoke' if SMOKE else 'full'} shape")

# =============================================================================
# P2 — the protocol rebuild (g1c's gates VERBATIM, e324's P0 path)
# =============================================================================
corpus = CharCorpus(ROOT / "data" / "input.txt", seed=1337)
stoi, itos = corpus.stoi, corpus.itos
vocab = corpus.vocab_size
zid = stoi["Z"]
train_ids, val_ids = corpus.train, corpus.val
train_text = "".join(itos[int(i)] for i in train_ids)
val_text = "".join(itos[int(i)] for i in val_ids)
G_VOCAB = {"vocab_size": vocab, "expected": 65, "pass": bool(vocab == 65)}
assert G_VOCAB["pass"], f"vocab drift: {vocab}"

corpus_counts = {"ZEPH": train_text.count("ZEPH"),
                 "TAVI": train_text.count("TAVI"),
                 "QELV": train_text.count("QELV"),
                 "NYST": train_text.count("NYST")}
G_NAMEFREE = {"corpus_counts": corpus_counts,
              "pass": bool(all(v == 0 for v in corpus_counts.values()))}
assert G_NAMEFREE["pass"], f"name-free gate FAILED: {G_NAMEFREE}"

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

name_ids = corpus.encode(G1.NAME)
name_ids_t = corpus.encode("TAVIREN")
name_ids_q = corpus.encode("QELVARO")
name_ids_n = corpus.encode("NYSTORA")
tid, qid, nid = stoi["T"], stoi["Q"], stoi["N"]
for nm, ids in (("ZEPHYRA", name_ids), ("TAVIREN", name_ids_t),
                ("QELVARO", name_ids_q), ("NYSTORA", name_ids_n)):
    assert len(ids) == 7, f"{nm} is not 7 chars"

bat_ids = {}
for j in G1.GEOS:
    cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
    bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
G_BATTERY = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape) for j in G1.GEOS},
             "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                          and list(bat_ids[0].shape) == [60, G1.PRE]
                          and list(bat_ids[12].shape)
                          == [60, G1.PRE + 12])}
assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]


def build_win(p, host, n_ids):
    return torch.cat([train_ids[p - G1.PRE: p], n_ids,
                      train_ids[p + len(host):
                                p + len(host) + G1.POST_CAP]])


win_bank = torch.stack([build_win(p, h, name_ids) for p, h in install_occ])
win_tav = torch.stack([build_win(p, h, name_ids_t) for p, h in install_occ])
win_qel = torch.stack([build_win(p, h, name_ids_q) for p, h in install_occ])
win_nys = torch.stack([build_win(p, h, name_ids_n) for p, h in install_occ])
win_masked_ok = all(
    "".join(itos[int(i)] for i in
            win_bank[i][G1.PRE: G1.PRE + 7]) == G1.NAME
    for i in range(60))
G_NAMEWIN = {"n_windows": 60, "masked_decode_all_name": bool(win_masked_ok),
             "parasites_differ_only_in_slot": bool(
                 win_bank[:, :G1.PRE].equal(win_tav[:, :G1.PRE])
                 and win_bank[:, G1.PRE + 7:].equal(win_tav[:, G1.PRE + 7:])
                 and win_bank[:, :G1.PRE].equal(win_qel[:, :G1.PRE])
                 and win_bank[:, G1.PRE + 7:].equal(win_qel[:, G1.PRE + 7:])
                 and win_bank[:, :G1.PRE].equal(win_nys[:, :G1.PRE])
                 and win_bank[:, G1.PRE + 7:].equal(win_nys[:, G1.PRE + 7:])),
             "pass": bool(win_masked_ok and list(win_bank.shape)
                          == [60, G1.BLOCK]
                          and win_bank[:, :G1.PRE].equal(win_tav[:, :G1.PRE]))}
assert G_NAMEWIN["pass"] and G_NAMEWIN["parasites_differ_only_in_slot"], \
    f"name-window gate FAILED: {G_NAMEWIN}"

anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                           for p, _ in install_occ])
inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
inst_mask[:, G1.PRE - 1: G1.PRE - 1 + 7] = True
G_INSTMASK = {"name_positions": int(inst_mask.sum()), "expected": 420,
              "pass": bool(int(inst_mask.sum()) == 420)}
assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

METRICS["gates"].update({"G_VOCAB": G_VOCAB, "G_NAMEFREE": G_NAMEFREE,
                         "G_SPLICE": G_SPLICE, "G_BATTERY": G_BATTERY,
                         "G_NAMEWIN": G_NAMEWIN, "G_INSTMASK": G_INSTMASK})
log("P2: protocol gates PASS (vocab 65 / namefree x4 / splice 19+41 / "
    "battery shapes / namewin + parasite-slot / install mask)")

# =============================================================================
# P3 — the pre-install states + the held room (bit-bound, certified)
# =============================================================================
base_net = G1.load_g1(GB.CKPT_DIR / "e001.pt")
base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
G_BASE = {"checkpoint": "runs/checkpoints/e001.pt",
          "params": base_net.num_params(),
          "fact_free_gm12": base_gm12,
          "pass": bool(base_gm12 <= 0.05
                       and base_net.num_params() == GB.G1B_PARAMS)}
assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
METRICS["gates"]["G_BASE"] = G_BASE
log(f"P3 G_BASE: e001 ({GB.G1B_PARAMS} params) fact-free gm12 "
    f"{base_gm12:.4f}: PASS")

root_net = G1.load_g1(GB.CKPT_DIR / "g1c_root.pt")     # ARMED (the rig's
theta_root = flat_params_cpu(root_net)                    # own convention)
root_read = G1.battery_cell(root_net, gm12_ids, zid)["mean_pz"]
G_ROOT = {"checkpoint": "runs/checkpoints/g1c_root.pt",
          "n_params": root_net.num_params(),
          "battery_read_measured": root_read,
          "battery_read_committed": E261.G1C_ROOT_GM12,
          "abs_diff": abs(root_read - E261.G1C_ROOT_GM12),
          "tol": E261.G_READ_TOL,
          "flat_md5": hashlib.md5(
              theta_root.numpy().tobytes()).hexdigest(),
          "pass": bool(root_net.num_params() == GB.G1B_PARAMS
                       and abs(root_read - E261.G1C_ROOT_GM12)
                       < E261.G_READ_TOL)}
assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
METRICS["gates"]["G_ROOT"] = G_ROOT
log(f"P3 G_ROOT: g1c_root {root_net.num_params()} params, battery read "
    f"{root_read:.10f} vs committed {E261.G1C_ROOT_GM12:.10f}: PASS")

N = root_net.num_params()
vmap_art = torch.load(GB.CKPT_DIR / "e258_vmap.pt", map_location="cpu",
                      weights_only=False)
v64_np = vmap_art["model"]["v_flat_fp32"].numpy().astype(np.float64)
span_art = torch.load(GB.CKPT_DIR / "e246_late_span.pt", map_location="cpu",
                      weights_only=False)
Vp = span_art["Vp"].contiguous()
G_BINDS = {
    "vmap": {"size": int(v64_np.size), "expected": N,
             "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
             "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                          and int(v64_np.size) == N)},
    "span": {"rank": int(Vp.shape[0]), "n": int(Vp.shape[1]),
             "md5": md5_of(GB.CKPT_DIR / "e246_late_span.pt"),
             "bound_md5": E261.E246_SPAN_MD5,
             "meta_experiment": span_art.get("meta", {}).get("experiment"),
             "pass": bool(md5_of(GB.CKPT_DIR / "e246_late_span.pt")
                          == E261.E246_SPAN_MD5
                          and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                          and int(Vp.shape[1]) == N)},
}
assert G_BINDS["vmap"]["pass"] and G_BINDS["span"]["pass"], f"{G_BINDS}"
METRICS["gates"]["G_VMBIND"] = G_BINDS["vmap"]
METRICS["gates"]["G_SPANBIND"] = G_BINDS["span"]

# the held room (the canon's own K10K) — LadderRooms from the e261 module
HELD_ROOM_SEEDS = (26113, 26114)
ROOM_K = 10_000
ROOM_MODE = "K10K"
E261.LADDER = ((ROOM_K, HELD_ROOM_SEEDS[0], HELD_ROOM_SEEDS[1]),)
E261.RUNG_NAMES = {ROOM_K: ROOM_MODE}
params_ref = list(G1.evl_load(base_sd).parameters())
rooms = E261.LadderRooms(N, E261.LADDER, v64_np,
                         Vp.numpy().astype(np.float64), params_ref, CPU)
cert = rooms.certify()
G_PROJ = {"form": f"the held rank-{ROOM_K} room certified (fp64 CPU, "
                  f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED})",
          "reads": cert, "pass": bool(cert["pass"])}
assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"
METRICS["gates"]["G_PROJ"] = G_PROJ
for nm, r in cert["per_rung"].items():
    log(f"P3 G_PROJ room {nm}: k {r['k']} idem "
        f"{r['idempotency_max']:.1e} kept2 {r['kept2_mean']:.6f} vs "
        f"{r['kept2_expect']:.6f}")

rooms264 = torch.load(GB.CKPT_DIR / "e264_rooms.pt", map_location="cpu",
                      weights_only=False)


def _to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


D_mine = rooms.rooms[ROOM_MODE].D
S_mine = rooms.rooms[ROOM_MODE].S
D264 = _to_np(rooms264["model"]["K10K"]["D_int8"])
S264 = _to_np(rooms264["model"]["K10K"]["S"])
seeds_equal = (tuple(int(s) for s in rooms264["model"]["K10K"]["seeds"])
               == HELD_ROOM_SEEDS)
D_bit = bool(np.array_equal(D_mine.astype(np.int8), D264.astype(np.int8)))
S_bit = bool(np.array_equal(S_mine.astype(np.int64), S264.astype(np.int64)))
G_ROOMHELD = {
    "form": "the room is THE CANON'S OWN: rebuilt from the registered "
            "seeds and BIT-BOUND against e264_rooms.pt's committed K10K "
            "record (the +-1 diagonal D AND the kept set S exactly equal)",
    "seeds_match": bool(seeds_equal), "D_bit_equal": D_bit,
    "S_bit_equal": S_bit,
    "pass": bool(seeds_equal and D_bit and S_bit),
}
assert G_ROOMHELD["pass"], f"held-room gate FAILED: {G_ROOMHELD}"
METRICS["gates"]["G_ROOMHELD"] = G_ROOMHELD
del rooms264, vmap_art, span_art, base_net
log("P3 G_ROOMHELD: the held room bit-bound to e264_rooms.pt's K10K: PASS")

# =============================================================================
# P4 — the step machinery (e261's step body VERBATIM, on CPU)
# =============================================================================
NAME_BS, CORP_BS, MIX_RANDOM, LR = (G1.NAME_BS, E43.CORP_BS,
                                    E43.MIX_RANDOM, E43.LR)
N_INST = win_bank.shape[0]
N_ANC = anchor_full.shape[0]


def first_batch(gen_seed: int):
    """The rig's step-1 Dmix draws (ix/aj/rj), exactly chunked_install's
    order: ix(16) install, aj(16) anchors, rj(32) random corpus."""
    gen = torch.Generator().manual_seed(int(gen_seed))
    ix = torch.randint(N_INST, (NAME_BS,), generator=gen)
    aj = torch.randint(N_ANC, (CORP_BS - MIX_RANDOM,), generator=gen)
    rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (MIX_RANDOM,),
                       generator=gen)
    return ix, aj, rj, gen


def run_first_step(net0, win_x, gen_seed, mode, zid_draw, return_d1=False):
    """ONE step of e261's chunked_install body VERBATIM on CPU + the
    geometry capture. The arithmetic lines are chunked_install's own
    (batch -> masked union CE -> clip -> the room hook -> opt.step); the
    captures read the same tensors the rig computes with."""
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    theta0 = flat_params_cpu(net).double().numpy().astype(np.float64)

    # ---- the read-gradient direction r at theta0 (the instrument) -------
    r_net = copy.deepcopy(net0)
    r_net.eval()
    r_net.zero_grad(set_to_none=True)
    n_ctx = g0_ids.shape[0]
    tot = None
    for i in range(0, n_ctx, 30):
        lg, _ = r_net(g0_ids[i:i + 30])
        lp = F.log_softmax(lg[:, -1], -1)[:, zid_draw]
        tot = lp.sum() if tot is None else tot + lp.sum()
    (tot / n_ctx).backward()
    r = torch.cat([p.grad.detach().reshape(-1)
                   for p in r_net.parameters()]).double().numpy() \
        .astype(np.float64)
    r_norm = float(np.linalg.norm(r))
    assert r_norm > 0, "read-gradient is zero"
    r_hat = r / r_norm
    r_net.zero_grad(set_to_none=True)
    del r_net

    # prior_cpu: the read at theta0 (the settled eval path)
    prior_net = copy.deepcopy(net0)
    prior_cpu = G1.battery_cell(prior_net, g0_ids, zid_draw)["mean_pz"]
    del prior_net

    # ---- the rig's step 1, VERBATIM --------------------------------------
    f = cosine_lr(0, E261.INST_TOTAL)               # house schedule (s-1=0)
    for g in opt.param_groups:
        g["lr"] = LR * f
    lr_s1 = LR * f
    ix, aj, rj, _gen = first_batch(gen_seed)
    corp = torch.cat([anchor_full[aj],
                      torch.stack([train_ids[s: s + G1.BLOCK]
                                   for s in rj])], 0)
    nw = win_x[ix]
    x = torch.cat([nw[:, :-1], corp[:, :-1]], 0).to(CPU)
    y = torch.cat([nw[:, 1:], corp[:, 1:]], 0).to(CPU)
    m = torch.zeros(NAME_BS + CORP_BS, x.shape[1], dtype=torch.bool)
    m[:NAME_BS] = inst_mask[ix]
    logits, _ = net(x)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          y.reshape(-1), reduction="none"
                          ).view(x.shape[0], x.shape[1])
    nm = nll[:NAME_BS][m[:NAME_BS]]
    cm = nll[NAME_BS:]
    loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    # ---- THE HOOK (verbatim; FREE = read-only) ---------------------------
    led = rooms.step_hook(list(net.parameters()), mode)
    g1 = torch.cat([p.grad.detach().reshape(-1)
                    for p in net.parameters()]).double().numpy() \
        .astype(np.float64)
    g1_norm = float(np.linalg.norm(g1))
    opt.step()
    theta1_raw = flat_params_cpu(net).double().numpy().astype(np.float64)

    # ---- the effective step: the eval path the committed s1 read saw -----
    sd_cpu = {k: v.detach().cpu().clone()
              for k, v in net.state_dict().items()}
    evl = copy.deepcopy(net0)
    evl.load_state_dict(sd_cpu)
    bcell = G1.battery_cell(evl, g0_ids, zid_draw)
    s1_cpu = bcell["mean_pz"]
    argmax = bcell["frac_argmax_z"]
    theta1_eff = flat_params_cpu(evl).double().numpy().astype(np.float64)
    del net, evl

    d1 = theta1_eff - theta0
    d1_raw = theta1_raw - theta0
    d1_norm = float(np.linalg.norm(d1))
    in_room = (float(np.linalg.norm(rooms.proj_of(mode, d1)) / d1_norm)
               if mode != "FREE" else 1.0)
    sign_neg = -np.sign(g1)
    cos_sign = float(d1 @ sign_neg
                     / (d1_norm * float(np.linalg.norm(sign_neg))))
    dose_read = float(d1 @ r_hat)
    cos_read = float(d1 @ r / (d1_norm * r_norm))
    dose_g = float(g1 @ r_hat)
    cos_g = float(g1 @ r / (g1_norm * r_norm))
    fp = {
        "gen": int(gen_seed), "mode": mode, "lr_s1": lr_s1,
        "loss_s1": float(loss.item()), "led": led,
        "theta0_md5": hashlib.md5(theta0.tobytes()).hexdigest(),
        "r_norm": r_norm, "prior_cpu": prior_cpu,
        "g1_norm": g1_norm, "kept_frac": led["kept_frac"],
        "d1_norm": d1_norm, "d1_raw_norm": float(np.linalg.norm(d1_raw)),
        "wall_correction": float(np.linalg.norm(d1_raw - d1)),
        "in_room_frac": in_room, "cos_sign_ray": cos_sign,
        "dose_read_step": dose_read, "cos_read_step": cos_read,
        "dose_read_grad": dose_g, "cos_read_grad": cos_g,
        "s1_cpu": s1_cpu, "s1_argmax_frac": argmax,
        "batch_ix_md5": hashlib.md5(ix.numpy().tobytes()).hexdigest(),
        "batch_aj_md5": hashlib.md5(aj.numpy().tobytes()).hexdigest(),
        "batch_rj_md5": hashlib.md5(rj.numpy().tobytes()).hexdigest(),
    }
    if return_d1:
        return fp, d1
    return fp


WINSETS = {"ZEPHYRA": win_bank, "TAVIREN": win_tav,
           "QELVARO": win_qel, "NYSTORA": win_nys}
ZIDS = {"ZEPHYRA": zid, "TAVIREN": tid, "QELVARO": qid, "NYSTORA": nid}

# THE NET0 MAP — each parent's ACTUAL convention (smoke catch 2's repair;
# verified against the committed step-1 ledger fingerprints: the canon's
# committed ce_batch 1.07017135... reproduces BIT-EXACTLY from the BASE;
# e324/GEN32401's committed ce_batch 1.11510777... reproduces BIT-EXACTLY
# from the ROOT):
#   canon 24314  -> BASE  (e264's run_arm passes G1.evl_load(base_sd))
#   e324/e327    -> ROOT  (their drivers pass load_g1(g1c_root.pt))
#   parasites    -> BASE  (e311/x28/e330 pass G1.evl_load(base_sd))
NET0_KIND = {"canon": "base", "e324": "root", "e327": "root",
             "e311": "base", "x28": "base", "e330": "base"}


def net0_for(row_draw: str):
    lineage = row_draw.split("/")[0]
    kind = NET0_KIND[lineage]
    return G1.evl_load(base_sd) if kind == "base" else root_net


# =============================================================================
# P5 — THE ATLAS (the fingerprint over the corpus + the rider)
# =============================================================================
atlas: list[dict] = []
for row in ROWS_ADJ:
    fp = run_first_step(net0_for(row["draw"]), WINSETS[row["name"]],
                        row["gen"], ROOM_MODE, ZIDS[row["name"]])
    dec = math.log10(max(fp["s1_cpu"], 1e-30)
                     / max(row["s1_committed"], 1e-30))
    # side_cpu on X34'S OWN AXIS (vs the committed prior basis, e001 for
    # the ZEPHYRA draws) so side_cpu means exactly what the committed
    # side means; the STANDING-read ratio (vs prior_cpu, the pre-install
    # organism's own read) is the discovery co-axis
    ratio_vs_committed = fp["s1_cpu"] / max(row["prior_committed"], 1e-30)
    ratio_standing = fp["s1_cpu"] / max(fp["prior_cpu"], 1e-30)
    ratio_committed = row["s1_committed"] / row["prior_committed"]
    side_cpu = ("SURGE" if ratio_vs_committed >= 100.0 else
                ("NO-SURGE" if ratio_vs_committed <= 1.2 else "AMBIGUOUS"))
    side_committed = "SURGE" if row["side"] == "SURVIVE" else "NO-SURGE"
    same_side = bool(side_cpu == side_committed)
    fp.update({
        "draw": row["draw"], "name": row["name"], "side": row["side"],
        "s1_committed": row["s1_committed"],
        "prior_committed": row["prior_committed"],
        "fidelity_decades": dec, "same_side": same_side,
        "ratio_vs_committed_prior": ratio_vs_committed,
        "ratio_to_standing": ratio_standing,
        "ratio_committed": ratio_committed,
        "standing_read": fp["prior_cpu"],
        "committed_ratio_to_standing": (row["s1_committed"]
                                        / max(fp["prior_cpu"], 1e-30)),
        "side_cpu": side_cpu,
    })
    atlas.append(fp)
    log(f"P5 {row['draw']:<14} gen {row['gen']:<10} [{row['side']:<12}] "
        f"s1_cpu {fp['s1_cpu']:.4e} (committed {row['s1_committed']:.4e}, "
        f"{dec:+.3f} dec; vs-standing {ratio_standing:.4f}) |d1| "
        f"{fp['d1_norm']:.4e} in-room {fp['in_room_frac']:.4f} cos_sign "
        f"{fp['cos_sign_ray']:.6f} dose {fp['dose_read_step']:+.3e} "
        f"cos_read {fp['cos_read_step']:+.3e}")

# the rider: the canon's first batch, NO projection (mode FREE)
canon_gen_actual = next(r["gen"] for r in ROWS if r["draw"] == "canon")
rider = run_first_step(net0_for("canon"), win_bank, canon_gen_actual,
                       "FREE", zid)
rider_dec = math.log10(max(rider["s1_cpu"], 1e-30)
                       / max(float(CANON_FREE_S1), 1e-30))
log(f"P5 RIDER canon@FREE: s1_cpu {rider['s1_cpu']:.4e} (committed "
    f"{float(CANON_FREE_S1):.4e}, {rider_dec:+.3f} dec) |d1| "
    f"{rider['d1_norm']:.4e} dose {rider['dose_read_step']:+.3e} kept "
    f"{rider['kept_frac']:.4f}")

# =============================================================================
# P6 — the separation adjudication (the frozen bars)
# =============================================================================
STAT_KEYS = [
    ("F1_d1_norm", "d1_norm"),
    ("F2_in_room_frac", "in_room_frac"),
    ("F3_cos_sign_ray", "cos_sign_ray"),
    ("F4_dose_read_step", "dose_read_step"),
    ("F5_cos_read_step", "cos_read_step"),
    ("F6_dose_read_grad", "dose_read_grad"),
    ("F7_cos_read_grad", "cos_read_grad"),
]


def separation(table, key):
    sur = [d[key] for d in table if d["side"] == "SURVIVE"]
    nos = [d[key] for d in table if d["side"] != "SURVIVE"]
    gap = None
    if max(sur) < min(nos):
        gap = ("surge_below", min(nos) - max(sur))
    elif max(nos) < min(sur):
        gap = ("surge_above", min(sur) - max(nos))
    return {"surge_range": [float(min(sur)), float(max(sur))],
            "nosurge_range": [float(min(nos)), float(max(nos))],
            "zero_overlap": gap is not None,
            "separation_dir": gap[0] if gap else None,
            "gap": float(gap[1]) if gap else None}


sep_all = {k: separation(atlas, key) for k, key in STAT_KEYS}
zeph_only = [d for d in atlas if d["name"] == "ZEPHYRA"]
sep_zeph = {k: separation(zeph_only, key) for k, key in STAT_KEYS}

hammers = [k for k, s in sep_all.items() if s["zero_overlap"]]
verdict = "ONE-HAMMER" if hammers else "THREE-DEATHS"

fid_fail = [d["draw"] for d in atlas
            if (abs(d["fidelity_decades"]) > 1.0 or not d["same_side"])]
G_FIDELITY = {
    "form": "per draw (strict birth form, restored after the net0 repair): "
            "the CPU-recomputed s1 lands within 1 decade of the committed "
            "s1 AND on the same side of the fork on X34'S AXIS (s1 vs the "
            "committed prior basis; surge >= 100x, no-surge <= 1.2x, the "
            "interior AMBIGUOUS, disclosed per draw). After the net0 "
            "repair every draw's step is recomputed from its parent's "
            "ACTUAL starting state — the canon from the BASE reproduces "
            "its committed record BIT-EXACTLY (ce and s1 identical to 16 "
            "digits); the strict gate is expected to pass for all 16.",
    "fidelity_decades": {d["draw"]: d["fidelity_decades"] for d in atlas},
    "same_side": {d["draw"]: d["same_side"] for d in atlas},
    "side_cpu": {d["draw"]: d["side_cpu"] for d in atlas},
    "rider_canon_free_decades": rider_dec,
    "failures": list(fid_fail),
    "pass": bool(not fid_fail),
}
METRICS["gates"]["G_FIDELITY"] = G_FIDELITY
log(f"P6 G_FIDELITY: {'PASS' if not fid_fail else 'FAIL ' + str(fid_fail)} "
    f"(rider FREE {rider_dec:+.3f} dec)")

# ---- P6b: the no-event probes (the t-scan + the sign-flip) --------------
def t_scan(net0, zid_draw, d1_vec, ts):
    """read(t) = the battery read on theta0 + t*d1 (the settled eval path)."""
    out = []
    theta0 = flat_params_cpu(net0).double()
    with torch.no_grad():
        for t in ts:
            th = theta0 + float(t) * torch.from_numpy(d1_vec)
            net_t = copy.deepcopy(net0)
            i = 0
            for p in net_t.parameters():
                n = p.numel()
                p.copy_(th[i:i + n].float().view(p.shape))
                i += n
            out.append((float(t),
                        G1.battery_cell(net_t, g0_ids, zid_draw)["mean_pz"]))
            del net_t
    return out


knife = {"form": "THE NO-EVENT PROBES (post-confound repair): the "
                 "step-scale t-scan read(t) on theta0 + t*d1 for the "
                 "canon (from the BASE, its true convention) + the FREE "
                 "rider + one root-start draw + one parasite — does ANY "
                 "first-step ray reach a read transition at up to 2x "
                 "scale? The sign-flip probe then asks the converse on "
                 "the canon (base start): can ANY sign pattern of the "
                 "first applied gradient ignite the read at step one? "
                 "Expected under the no-event picture: nothing moves at "
                 "any probed scale or pattern."}


def probe_knife(draw_row, mode):
    fp, d1_vec = run_first_step(net0_for(draw_row["draw"]),
                                WINSETS[draw_row["name"]], draw_row["gen"],
                                mode, ZIDS[draw_row["name"]],
                                return_d1=True)
    net0_ref = net0_for(draw_row["draw"])
    ts = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.5, 2.0]
    scan = t_scan(net0_ref, ZIDS[draw_row["name"]], d1_vec, ts)
    # t* := the SMALLEST t where the read CRASHES below 0.15 (leaves the
    # standing plateau DOWNWARD — the kill event); None if no crash by t=2;
    # NOT APPLICABLE when the standing read itself is below 0.15 (the
    # parasite rows — their plateau sits under the threshold)
    r2 = dict(scan).get(2.0)
    r0 = dict(scan).get(0.0)
    t_star = None
    if r2 is not None and r2 < 0.15 and r0 is not None and r0 >= 0.15:
        lo_t, hi_t = 0.0, 2.0        # read(lo)=high, read(hi)=crashed
        for _ in range(18):
            mid = (lo_t + hi_t) / 2
            r_mid = t_scan(net0_ref, ZIDS[draw_row["name"]], d1_vec,
                           [mid])[0][1]
            if r_mid >= 0.15:
                lo_t = mid
            else:
                hi_t = mid
        t_star = (lo_t + hi_t) / 2
    return {"draw": draw_row["draw"], "mode": mode,
            "s1_cpu": fp["s1_cpu"],
            "s1_committed": draw_row["s1_committed"],
            "dose_read_step": fp["dose_read_step"],
            "scan": scan, "t_star_kill": t_star}


canon_row = next(r for r in ROWS if r["draw"] == "canon")
_root_standing_meas = G1.battery_cell(root_net, g0_ids, zid)["mean_pz"]
G_STANDING = {
    "form": "the confound's own cross-check: the root's measured g0 "
            "battery read (the STANDING read that made root-start draws "
            "read ~0.745 at s1) vs e261's committed G1C_ROOT_G0 landing "
            "anchor",
    "measured": float(_root_standing_meas),
    "committed": E261.G1C_ROOT_G0,
    "abs_diff": abs(float(_root_standing_meas) - E261.G1C_ROOT_G0),
    "pass": bool(abs(float(_root_standing_meas) - E261.G1C_ROOT_G0)
                 <= 1e-12),
}
assert G_STANDING["pass"], "standing-read cross-check FAILED"
METRICS["gates"]["G_STANDING"] = G_STANDING
log(f"P6 G_STANDING: the root's standing g0 {_root_standing_meas:.10f} "
    f"== e261's committed G1C_ROOT_G0: PASS")
surge_row = next((r for r in ROWS if r["draw"] == "e324/32401"), None)
parasite_row = next(r for r in ROWS if r["name"] == "TAVIREN")
knife["probes"] = []
for prow, pmode in ((canon_row, ROOM_MODE), (canon_row, "FREE"),
                    (surge_row, ROOM_MODE) if surge_row else None,
                    (parasite_row, ROOM_MODE)):
    if prow is None:
        continue
    pk = probe_knife(prow, pmode)
    knife["probes"].append(pk)
    scan_txt = ", ".join(f"t={t:.3f}:{v:.4g}" for t, v in pk["scan"][:11])
    log(f"P6b knife {pk['draw']}@{pk['mode']}: t*_kill "
        f"{pk['t_star_kill']}; scan {scan_txt}")
knife["the_reading"] = (
    "Under the no-event picture every probed ray should stay on its "
    "state's own read level (the base's prior for base-start draws; the "
    "root's standing read for root-start draws) at every t up to 2 — "
    "no transition, in either direction, at any doubled-step scale. "
    "Any observed transition would locate a REAL step-one-reachable "
    "cliff and would be reported as such.")
METRICS["the_no_event_probes"] = knife

# ---- P6b2: the sign-flip probe (can anything ignite the read at step 1?) -
def probe_signflip(gen_seed, thresholds):
    """For each threshold: rerun the canon's first step VERBATIM (from
    the BASE, its true convention) but with the applied gradient's
    smallest-magnitude coordinates' signs FLIPPED (|g1| < thr -> -g1).
    Upper bound on pattern perturbation of the first applied gradient:
    if even near-total sign randomization cannot move the read off its
    prior at step one, the no-event is PATTERN-robust, not just
    magnitude-robust."""
    rows = []
    net0 = net0_for("canon")
    theta0 = flat_params_cpu(net0).double().numpy().astype(np.float64)
    for thr in thresholds:
        net = copy.deepcopy(net0).to(CPU)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1)
        f = cosine_lr(0, E261.INST_TOTAL)
        for g in opt.param_groups:
            g["lr"] = LR * f
        ix, aj, rj, _gen = first_batch(gen_seed)
        corp = torch.cat([anchor_full[aj],
                          torch.stack([train_ids[s: s + G1.BLOCK]
                                       for s in rj])], 0)
        nw = win_bank[ix]
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0).to(CPU)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0).to(CPU)
        m = torch.zeros(NAME_BS + CORP_BS, x.shape[1], dtype=torch.bool)
        m[:NAME_BS] = inst_mask[ix]
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        rooms.step_hook(list(net.parameters()), ROOM_MODE)
        with torch.no_grad():
            flipped = 0
            for p in net.parameters():
                gflat = p.grad.reshape(-1)
                sel = gflat.abs() < thr
                flipped += int(sel.sum())
                gflat[sel] = -gflat[sel]
        opt.step()
        sd_cpu = {k: v.detach().cpu().clone()
                  for k, v in net.state_dict().items()}
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd_cpu)
        read = G1.battery_cell(evl, g0_ids, zid)["mean_pz"]
        del net, evl
        rows.append({"threshold": float(thr),
                     "n_flipped": flipped,
                     "frac_flipped": flipped / N,
                     "read_after": float(read)})
        log(f"P6b2 signflip thr={thr:.1e}: flipped {flipped} coords "
            f"({flipped / N:.4%}); read {read:.4e}")
    return rows


knife["signflip"] = {
    "form": "the ignition probe: the canon's first step (from the BASE, "
            "its true convention) rerun with the applied gradient's "
            "|g1| < thr coordinates' signs flipped — near-total sign "
            "randomization of the first applied gradient; if the read "
            "stays at prior under every threshold, the step-one no-event "
            "is pattern-robust",
    "rows": probe_signflip(canon_gen_actual,
                           [1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]),
}

# ---- P6c: THE NET0 CONFOUND (the cell's discovery; numbers at runtime) --
canon_atlas = next(d for d in atlas if d["draw"] == "canon") \
    if any(d["draw"] == "canon" for d in atlas) else None
root_standing = G1.battery_cell(root_net, g0_ids, zid)["mean_pz"]
_root_stay = [d["s1_committed"] / root_standing for d in atlas
              if d["side"] == "SURVIVE" and d["name"] == "ZEPHYRA"]
_base_flat_rows = [d for d in atlas if NET0_KIND[d["draw"].split("/")[0]]
                   == "base"]
confound = {
    "form": "THE NET0 CONFOUND (smoke catch 2): the fork corpus mixed "
            "STARTING STATES. The committed canon (e264 K10K, resumed "
            "from e261's cut vehicle) and e272's K10KR replicate ran "
            "chunked_install from G1.evl_load(base_sd) — THE BASE e001; "
            "e324's and e327's census draws ran it from the g1c ROOT "
            "(load_g1) — the FORMED organism. The root's own standing g0 "
            "read (committed as e261's G1C_ROOT_G0) IS the 'surge': the "
            "root-start draws' s1 ~ the standing read (ratio-to-standing "
            "~ 1.0002), which x34's e001-prior axis read as ~5.5e4x "
            "'SURGE'; the base-start draws' s1 = prior x 1.002-1.006 "
            "('NO-SURGE'). AT STEP ONE NOTHING EVER HAPPENS — every "
            "draw's first step is a no-event on its own state's read. "
            "The s1 fork of x34/e324/e327 is (almost entirely) the "
            "starting-state confound; the [0.01, 0.15] 'empty gap' is "
            "the gap between the base's prior and the root's standing "
            "read — no step-one read can land there because step-one "
            "reads do not move. The canon's 'die-then-recover' texture "
            "re-words to plain FORMATION from the base (nothing -> 0.31 "
            "at s100 -> 0.2646 at s400). Verified at the fingerprint "
            "level: the canon's committed ce_batch reproduces "
            "BIT-EXACTLY from the base; GEN32401's from the root.",
    "root_standing_g0": float(root_standing),
    "root_standing_g0_committed": E261.G1C_ROOT_G0,
    "root_start_draws_committed_stay_ratio_range": (
        [float(min(_root_stay)), float(max(_root_stay))]
        if _root_stay else None),
    "base_start_draws": [d["draw"] for d in _base_flat_rows],
    "class_boundary_equals_net0_boundary": bool(
        all((d["side"] == "SURVIVE")
            == (NET0_KIND[d["draw"].split("/")[0]] == "root")
            for d in atlas)),
    "the_re_wording": (
        "ONE step-one texture exists: NO-EVENT (the read does not move "
        "at step one, for any draw, any name, either starting state, "
        "any room, either platform). The fork's discriminative content "
        "moves ENTIRELY to the formation phase (s100/s400/texture "
        "axes)."),
}
METRICS["the_net0_confound"] = confound
log(f"P6c NET0 CONFOUND: the root's standing g0 {root_standing:.10f} "
    f"(committed {E261.G1C_ROOT_G0:.10f}); root-start draws' committed "
    f"stay ratios {confound['root_start_draws_committed_stay_ratio_range']}"
    f"; class boundary == net0 boundary: "
    f"{confound['class_boundary_equals_net0_boundary']}")

# ---- DISCLOSURE: the class boundary == the net0 boundary ----------------
# (canon + parasites start from the base = the NO-SURGE class; e324/e327
# start from the root = the SURVIVE class) — any zero-overlap "hammer" in
# the frozen adjudication therefore separates STARTING STATES, not
# step-one textures; recorded in the_net0_confound.

# =============================================================================
# P7 — P-e328x32: the concrete registration (data-derived, frozen verbatim)
# =============================================================================
if not SMOKE:
    if verdict == "ONE-HAMMER":
        hammer = hammers[0]
        hs = sep_all[hammer]
        call = "FLIPS-TO-SURGE"
        concrete = (
            f"The atlas's verdict is ONE-HAMMER: {hammer} separates surge "
            f"from no-surge across all {len(atlas)} draws at zero overlap "
            f"(surge [{hs['surge_range'][0]:.6g}, {hs['surge_range'][1]:.6g}]"
            f" vs no-surge [{hs['nosurge_range'][0]:.6g}, "
            f"{hs['nosurge_range'][1]:.6g}], gap {hs['gap']:.6g}, "
            f"{hs['separation_dir']}). The first batch's content CARRIES "
            f"the surge geometry. Per the frozen mapping: fork-in-the-"
            f"batch — e328's swap arm (the canon's gen with 32401's first "
            f"minibatch) is predicted to FLIP TO SURGE: s1/prior >= 1e3 "
            f"(the FLIPS-TO-SURGE bar), with s1 expected inside the surge "
            f"draws' committed band (0.7449-0.7453). The control arm "
            f"(canon verbatim) is predicted to stay no-surge at its "
            f"committed 1.346643e-05.")
    else:
        parts = []
        for k, s in sep_all.items():
            parts.append(f"{k}: surge [{s['surge_range'][0]:.3g},"
                         f"{s['surge_range'][1]:.3g}] vs no-surge "
                         f"[{s['nosurge_range'][0]:.3g},"
                         f"{s['nosurge_range'][1]:.3g}] (overlap)")
        call = "STAYS-NO-SURGE"
        concrete = (
            "The atlas's verdict is THREE-DEATHS: no candidate statistic "
            "separates the textures at zero overlap — " + "; ".join(parts)
            + ". The first batch's content carries NO distinguishing "
            "geometry — and the net0 confound found on the way explains "
            "why nothing could: at step one NO draw's read moves at all "
            "(every first step is a no-event on its own state's prior; "
            "the fork itself was the starting-state confound). Per the "
            "frozen mapping: fork-in-the-gen — e328's swap arm (the "
            "canon's run from the BASE with 32401's first minibatch) is "
            "predicted to STAY NO-SURGE (s1/prior ~ 1.0x, the "
            "STAYS-NO-SURGE bar), matching the control arm's committed "
            "1.346643e-05; the FLIPS-TO-SURGE bar cannot fire under the "
            "corrected reading, and the fork-in-batch vs fork-in-gen "
            "content moves to the formation axes (s100/s400, write "
            "norm, in-room).")
    P_E328X32 = {
        "registered_at": now_iso(),
        "form": "P-e328x32 — x32's atlas answer APPLIED to e328's design, "
                "registered BEFORE e328's compute (the dispatch's "
                "ordering); the mapping rule was frozen at x32's BIRTH "
                "commit (this file's docstring), the concrete call below "
                "is data-derived at COMPLETE and never shopped",
        "atlas_verdict": verdict,
        "mapping_rule_verbatim": (
            "ONE-HAMMER -> FLIPS-TO-SURGE (the first batch carries a "
            "geometry that separates the textures; e328's swap arm "
            "predicts s1/prior >= 1e3); THREE-DEATHS -> STAYS-NO-SURGE "
            "(the first batch carries no distinguishing geometry; "
            "e328's swap arm predicts s1/prior ~ 1)"),
        "the_concrete_call": call,
        "the_concrete_call_text": concrete,
        "countervailing": (
            "P-e328a's own lean is STAYS-NO-SURGE weakly (e327's "
            "gen-tracked evidence). THE CONFOUND SHARPENER from this "
            "run: the fork corpus mixed starting states (the canon from "
            "the BASE, the 'surge' draws from the FORMED ROOT), and AT "
            "STEP ONE NOTHING EVER HAPPENS on either state's own read — "
            "so e328's swap arm (the canon's run from the BASE with "
            "32401's first minibatch) is predicted to read ~prior at s1 "
            "not because the gen carries the fork but because NO first "
            "page moves the read at all; the FLIPS-TO-SURGE bar cannot "
            "fire under the corrected reading, and the informative "
            "content moves ENTIRELY to the formation axes (s100/s400, "
            "write norm, in-room: does the swapped first page shift the "
            "canon's formation trajectory toward 32401's class or not — "
            "the fork-in-batch question now lives there, and e328's "
            "control arm re-verifies the canon's committed record "
            "bit-exactly on GPU)."),
    }
    METRICS["P_e328x32"] = P_E328X32
    log(f"P7 P-e328x32 REGISTERED: {call} (atlas {verdict})")

# =============================================================================
# P8 — the figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0))
ax = axes[0]
sur = [d for d in atlas if d["side"] == "SURVIVE"]
nos = [d for d in atlas if d["side"] != "SURVIVE"]
ax.scatter([d["dose_read_step"] for d in sur],
           [d["cos_read_step"] for d in sur],
           s=120, marker="o", color="#1f77b4", edgecolors="black",
           linewidths=0.6, label=f"SURGE (n={len(sur)})", zorder=3)
ax.scatter([d["dose_read_step"] for d in nos],
           [d["cos_read_step"] for d in nos],
           s=200, marker="D", color="#d62728", edgecolors="black",
           linewidths=0.6, label=f"NO-SURGE (n={len(nos)})", zorder=4)
for d in nos:
    ax.annotate(d["draw"], (d["dose_read_step"], d["cos_read_step"]),
                textcoords="offset points", xytext=(7, 5), fontsize=8)
if sur:
    ax.annotate("surge draws", (sur[0]["dose_read_step"],
                                sur[0]["cos_read_step"]),
                textcoords="offset points", xytext=(7, 5), fontsize=8)
ax.set_xlabel("F4 dose_read(d1) = d1.r/||r||  (nats, first-order)")
ax.set_ylabel("F5 cos(d1, r)")
ax.set_title("x32 — THE FIRST-STEP ATLAS: the read-direction dose plane\n"
             f"{len(atlas)} draws — verdict: "
             f"{verdict if not SMOKE else 'SMOKE (not adjudicated)'}")
ax.grid(alpha=0.25)
ax.legend()

ax2 = axes[1]
ypos = np.arange(len(STAT_KEYS))
for i, (k, key) in enumerate(STAT_KEYS):
    s = sep_all[k]
    ax2.plot([s["surge_range"][0], s["surge_range"][1]],
             [ypos[i] - 0.18, ypos[i] - 0.18], color="#1f77b4", lw=3,
             solid_capstyle="butt")
    ax2.plot([s["nosurge_range"][0], s["nosurge_range"][1]],
             [ypos[i], ypos[i]], color="#d62728", lw=3,
             solid_capstyle="butt")
    if s["zero_overlap"]:
        ax2.text(1.02, ypos[i], "ZERO-OVERLAP",
                 transform=ax2.get_yaxis_transform(), fontsize=8,
                 color="green", va="center")
ax2.set_yticks(ypos)
ax2.set_yticklabels([k for k, _ in STAT_KEYS], fontsize=9)
ax2.set_xlabel("statistic range (blue = surge, red = no-surge; linear "
               "axis — read the table for scales)")
ax2.set_title("the separation table (per frozen candidate)")
ax2.grid(alpha=0.25, axis="x")
fig.tight_layout()
png = OUT / "x32_first_step_atlas.png"
fig.savefig(png, dpi=130)
log(f"P8: figure -> {png.relative_to(ROOT)}")

# =============================================================================
# P9 — REPORT.md + metrics out
# =============================================================================
lines = []
lines.append("# x32 — THE FIRST-STEP ATLAS\n")
lines.append(f"Run: {now_iso()} ({'SMOKE' if SMOKE else 'FULL'}) — pure "
             "CPU; step one per draw recomputed with e261's step body "
             "VERBATIM; every artifact md5-bound; every label read at "
             "runtime from x34's census.\n")
lines.append("\n## THE ATLAS TABLE (one row per draw)\n")
lines.append("| draw | gen | side | s1 committed | s1 cpu | fid (dec) | "
             "cpu/standing | committed/standing | d1 norm | in-room | "
             "cos sign-ray | dose read (d1) | cos(d1,r) | kept |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in atlas:
    lines.append(
        f"| {d['draw']} | {d['gen']} | {d['side']} | "
        f"{d['s1_committed']:.4e} | {d['s1_cpu']:.4e} | "
        f"{d['fidelity_decades']:+.3f} | {d['ratio_to_standing']:.6f} | "
        f"{d['committed_ratio_to_standing']:.3e} | "
        f"{d['d1_norm']:.4e} | {d['in_room_frac']:.4f} | "
        f"{d['cos_sign_ray']:.6f} | {d['dose_read_step']:+.3e} | "
        f"{d['cos_read_step']:+.3e} | {d['kept_frac']:.4f} |")
lines.append("\nRIDER (excluded from adjudication): the canon's first "
             f"batch, mode FREE — s1_cpu {rider['s1_cpu']:.4e} vs "
             f"committed {float(CANON_FREE_S1):.4e} ({rider_dec:+.3f} "
             f"dec); |d1| {rider['d1_norm']:.4e}; dose_read "
             f"{rider['dose_read_step']:+.3e}; kept_frac "
             f"{rider['kept_frac']:.4f}.\n")
lines.append("\n## THE SEPARATION TABLE (the frozen bars)\n")
lines.append("| statistic | surge range | no-surge range | zero overlap? "
             "| ZEPHYRA-only zero overlap? |")
lines.append("|---|---|---|---|---|")
for k, _ in STAT_KEYS:
    s, sz = sep_all[k], sep_zeph[k]
    lines.append(
        f"| {k} | [{s['surge_range'][0]:.6g}, {s['surge_range'][1]:.6g}] | "
        f"[{s['nosurge_range'][0]:.6g}, {s['nosurge_range'][1]:.6g}] | "
        f"{'YES' if s['zero_overlap'] else 'no'} | "
        f"{'YES' if sz['zero_overlap'] else 'no'} |")
verdict_str = verdict if not SMOKE else "SMOKE — not adjudicated"
lines.append(f"\n**VERDICT: {verdict_str}.**"
             + (f" Hammer(s): {hammers}." if hammers else ""))
lines.append(
    "\nVERDICT INTERPRETATION (the confound clause, mandatory reading): "
    "the frozen class boundary (SURVIVE vs DIE) coincides with the NET0 "
    "boundary (root-start vs base-start) — ANY zero-overlap hammer above "
    "separates STARTING STATES, not step-one textures. Under the "
    "corrected reading there is ONE step-one texture: the NO-EVENT. "
    "The fork's discriminative content lives in the formation phase.\n")
lines.append("\n## THE NET0 CONFOUND (the cell's discovery)\n")
lines.append(f"- {confound['form']}")
lines.append(f"- The root's standing g0 read: {confound['root_standing_g0']:.10f} "
             f"(== e261's committed G1C_ROOT_G0, gate G_STANDING).")
lines.append(f"- Root-start draws' committed s1 / standing: "
             f"{confound['root_start_draws_committed_stay_ratio_range']} "
             f"— the 'surge' is the standing read STAYING.")
lines.append(f"- Base-start draws: {confound['base_start_draws']} — "
             f"s1 = prior x 1.002-1.006.")
lines.append(f"- Class boundary == net0 boundary: "
             f"{confound['class_boundary_equals_net0_boundary']}.")
lines.append(f"- {confound['the_re_wording']}\n")
lines.append("\n## THE NO-EVENT PROBES\n")
for pk in knife.get("probes", []):
    scan_txt = " ".join(f"t={t:.3f}:{v:.3g}" for t, v in pk["scan"][:11])
    lines.append(f"- {pk['draw']}@{pk['mode']} (from "
                 f"{NET0_KIND[pk['draw'].split('/')[0]]}): s1_cpu "
                 f"{pk['s1_cpu']:.4e} vs committed "
                 f"{pk['s1_committed']:.4e}; transition t* = "
                 f"{pk['t_star_kill']}; dose_read "
                 f"{pk['dose_read_step']:+.3e}")
    lines.append(f"  - scan: {scan_txt}")
lines.append(f"\n{knife['the_reading']}\n")
sf = knife.get("signflip")
if sf:
    lines.append("\n### the sign-flip probe (can any pattern ignite the "
                 "read at step one — the canon from the BASE)\n")
    lines.append("| flip threshold | coords flipped | fraction | read "
                 "after step |")
    lines.append("|---|---|---|---|")
    for r in sf["rows"]:
        lines.append(f"| |g1| < {r['threshold']:.1e} | {r['n_flipped']} | "
                     f"{r['frac_flipped']:.4%} | {r['read_after']:.4e} |")
    lines.append("")
if hammers and any(k.startswith("F3_") for k in hammers):
    lines.append("\nRECIPE-IDENTITY CAVEAT: a hammer on F3 (the sign-ray) "
                 "is pinned by AdamW's construction (e231) and carries "
                 "the instrument warning.\n")
if not SMOKE:
    lines.append("\n## P-e328x32 (registered BEFORE e328's compute)\n")
    lines.append(f"- Mapping rule (frozen at birth): "
                 f"{P_E328X32['mapping_rule_verbatim']}")
    lines.append(f"- **THE CONCRETE CALL: "
                 f"{P_E328X32['the_concrete_call']}**")
    lines.append(f"- {P_E328X32['the_concrete_call_text']}")
    lines.append(f"- Countervailing: {P_E328X32['countervailing']}\n")
lines.append("\n## DISCLOSURES\n")
lines.append("- NET0 (the confound repair): each draw's step is "
             "recomputed from its parent's ACTUAL starting state — the "
             "canon + parasites from the BASE e001 (e264/e272/e311/x28/"
             "e330's convention), e324/e327's draws from the g1c ROOT "
             "(their convention). The canon's committed record "
             "reproduces BIT-EXACTLY from the base (ce and s1 to 16 "
             "digits); GEN32401's from the root.")
lines.append("- Platform: the committed parents ran the step on GPU fp32; "
             "this atlas recomputes it on CPU fp32 for ALL draws on ONE "
             "platform. The fidelity gate (same side of the fork, within 1 "
             "decade; exact to fp32 round-off in practice) validates "
             "every draw; the GEOMETRY (the datum) is internally "
             "consistent by construction.")
lines.append("- The effective step d1 includes the eval path's wall "
             "settle (the committed s1 was read on the settled state); "
             "the raw post-opt step and the wall correction are "
             "co-reported per draw in metrics.json. For the base-start "
             "draws the organism is UNANCHORED (evl_load disarms) so the "
             "wall correction is exactly zero; for root-start draws the "
             "root's own anchors settle.")
lines.append("- F3 is recipe-pinned (e231): AdamW's first displacement "
             "is -lr(sign(g)+wd*theta) by construction; it is reported "
             "for the law's cross-check, not offered as a mechanism.")
lines.append("- The parasite draws are CROSS-NAME rows (different names, "
             "different priors); the ZEPHYRA-only separation columns "
             "carry the matched-name comparison.")
lines.append("- x34's e001-prior basis for the ZEPHYRA draws is retained "
             "in the frozen labels (as registered); the net0 confound "
             "section carries the corrected per-state basis.\n")
gp = {k: v.get("pass", True) for k, v in METRICS["gates"].items()}
METRICS["gates_summary"] = {"n_gate_classes": len(gp),
                            "n_pass": sum(1 for v in gp.values() if v),
                            "detail": gp}
lines.append("\n## GATES\n")
lines.append(f"{METRICS['gates_summary']['n_pass']}/"
             f"{METRICS['gates_summary']['n_gate_classes']} gate classes "
             "PASS (G_PARENTS md5; the protocol gates; G_ROOT/G_BASE; "
             "G_VMBIND/G_SPANBIND; G_PROJ; G_ROOMHELD bit-bind; "
             "G_FIDELITY per-draw fork fidelity).\n")
METRICS["the_atlas"] = atlas
METRICS["the_rider_free"] = dict(rider)
METRICS["the_rider_free"]["committed_s1"] = float(CANON_FREE_S1)
METRICS["the_rider_free"]["fidelity_decades"] = rider_dec
METRICS["separation_all16"] = sep_all
METRICS["separation_zephyra13"] = sep_zeph
METRICS["verdict"] = verdict if not SMOKE else "SMOKE-NOT-ADJUDICATED"
METRICS["phase"] = "SMOKE-COMPLETE" if SMOKE else "COMPLETE"
METRICS["date_completed"] = now_iso()
METRICS["outputs"] = [str((OUT / n).relative_to(ROOT)) for n in
                      ("metrics.json", "REPORT.md",
                       "x32_first_step_atlas.png")]
report = OUT / "REPORT.md"
report.write_text("\n".join(lines), encoding="utf-8")
with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(METRICS, f, indent=1, default=str)
log(f"P9: REPORT + metrics -> {OUT.relative_to(ROOT)}")

gs = METRICS["gates_summary"]
if not SMOKE:
    assert gs["n_pass"] == gs["n_gate_classes"], \
        f"gate failures: {gs['detail']}"
log(f"DONE ({'SMOKE' if SMOKE else 'FULL'}): {len(atlas)} draws, verdict "
    f"{METRICS['verdict']}, gates {gs['n_pass']}/{gs['n_gate_classes']}")
