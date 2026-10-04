"""E257 — T231's registered sharper prediction: THE SUPPORT-V-OVERLAP READ
(the composition's probe-level discriminator; dispatched 2026-10-04;
desk/CPU, e248 owns the GPU).

THE QUESTION (verbatim from the dispatch): does a probe's SUPPORT-V-OVERLAP
(its support's share of v's mass — the direct denominator load on its own
direction) predict its thermal share BETTER than the span-residency did
(rho -0.148)? If yes: the composition's carrier claim gains its probe-level
evidence; if no: the coupling's carrier is the span-basis specifically, not
v-load generally.

BARS VERBATIM (frozen from the dispatch BEFORE this script or any compute;
adjudicate against exactly this; no bar shopping):
  - V-OVERLAP-WINS — "the support-v-overlap join's |rho| exceeds the
    span-residency join's by >= 0.05 pooled (both computed here on
    identical probes) — the denominator load on the probe's OWN direction
    is the better carrier; the composition's probe-level evidence landed"
  - SPAN-SPECIFIC — "the v-overlap join does NOT beat the span join
    (within 0.05) — the carrier is the span basis specifically; the
    composition bounded"
  - MIXED — "the tables verbatim, both joins, per wash"

OPERATIONALIZATIONS (frozen here BEFORE compute; they fix the clauses, they
do not move the bars):
  * THE X-SIDE (the support-v-overlap): qov_i(w, s) = <s_i, V_s s_i> /
    ||s_i||^2 — the local quadratic form of the reconstructed AdamW second
    moment at state s on the probe's OWN t=0 support direction (the FULL
    form, not a low-rank approximation: v is reconstructed exactly, fp64,
    and the form is computed chunk-wise over all 124,439,808 coordinates;
    the fp16 support-cache floor rides it, disclosed). s_i = e226/e239's
    committed t=0 supports (rebuilt + certified); V_s = e240's fp64
    recursion over the certified wash replay, PRIMARY flavor = CLIPPED
    gradients (what the wash's AdamW drank — e240/e254's primary); the
    RAW-gradient flavor v2 co-reported. States = exactly e238's committed
    residual-cell states {w1, w2} x {50, 80} (the y-side's own states; w3
    has no residual cells — the join scope IS the y-side's scope, and the
    span join is recomputed on the IDENTICAL rows). t=0 disclosed: v_0 = 0
    exactly (fresh optimizer state; e254's disclosure) — the t=0 overlap
    is trivially 0 and carries no rank information.
  * THE Y-SIDE (frozen, committed): e238's per-probe thermal deviation
    |resid_z| — the within-battery z of the one-T model's residual — read
    at runtime from e238's committed residual cells (16 cells = {w1, w2} x
    {50, 80} x 4 batteries; 216 probe-rows), EXACTLY e256's join-3 row
    construction (same cell order, same canonical battery order).
  * THE JOIN: Spearman(qov, |resid_z|) — PRIMARY pooled over the identical
    216 rows; co-reports per wash (108 rows each), per cell (4 x 54), the
    signed-resid_z flavor, the v2 (raw-v) flavor.
  * THE COMPARISON (the registered bar's arithmetic): the span-residency
    join RECOMPUTED HERE on the identical 216 rows from e256's committed
    census (residency = split-A primary, the carrier of the committed
    rho -0.148), certified in G_ROWS against e256's committed rho
    (-0.14793697753319948, tol 1e-12). delta_pool := |rho_v_pool| -
    |rho_span_pool|; delta_w := the same per wash (w1/w2, the identical
    108-row subsets).
  * ADJUDICATION (frozen; order V-OVERLAP-WINS -> SPAN-SPECIFIC -> MIXED;
    the letter of the pooled clause decides, MIXED is the wash-discordance
    contingency, mirroring e256's discordance rule):
      discordant := (delta_pool >= 0.05 AND min(delta_w1, delta_w2)
                     <= -0.05) OR (delta_pool < 0.05 AND
                     min(delta_w1, delta_w2) >= 0.05)
      V-OVERLAP-WINS := delta_pool >= 0.05 AND NOT discordant
      SPAN-SPECIFIC  := delta_pool <  0.05 AND NOT discordant
      MIXED          := discordant
    (a wash "hard-contradicts" a pooled win only at the bar's own 0.05
    margin in the opposite direction; ties are not contradictions. No bar
    shopping: the letter of the dispatch clauses + this frozen rule.)
  * CO-READ 1 (the anchors): Gmail/iPhone qov z vs the product family's
    5 non-anchor probes (e239's band convention), per (wash, state) —
    read against e256's committed anchor residency zs.
  * CO-READ 2 (nearrel vs product): the families' qov medians per
    (wash, state); the near-uscap/product ratio — read against e256's
    committed t=0 residency ratio (the "nearrel stands furthest from the
    span" finding).
  * CO-READ 3 (the base-rate disclosure): per (wash, state) — mean(v)
    (= ||v||_1/N, the Haar E[<u, V u>] for a random unit direction), the
    random-direction band from N_NULL fresh gaussian unit draws
    (exact per-draw normalization; disclosed seeds; fp32 generation,
    fp64 algebra), the 54 probes' qov distribution vs that band, and v's
    participation ratio (||v||_1^2/||v||_2^2 — how concentrated the
    denominator is). A large probe-overlap/base-rate ratio is
    arithmetically pre-announced (v is built from squared gradients and
    the supports ARE gradient directions of the same organism) — the
    band is the disclosure, the JOIN is the registered read; no bar
    speaks on the enrichment.
  * MULTIPLICITY: one primary join + registered co-report flavors
    (signed, v2, per-wash, per-cell); NO correction claimed (disclosed).

GATES: G_ENV (desk-only CPU: no GPU, no CUDA calls; torch threads 8 =
e182c's bit-exact replay convention VERBATIM from e240/e254 — the
dispatch's threads-4 clause is honored on the numpy/BLAS side, exactly as
e254's G_ENV under the same dispatch regime; psutil load checks per phase;
RAM wait-floors), G_SIZE (inherited 124M-with-reason), G_CORPUS, G_BATT
(e240's battery build, e254-verbatim), G_ORDER (my 54 == e226's committed
54: facts, batteries, family6, order — the join's row alignment),
G_SUPPORT (FD + the e226 w1:s0 alignment certification, dp <= 0.005),
G_CE (bit-exact CE + raw gnorm vs e234's committed journal, every step),
G_DRAWS (bit-exact generator state vs the archived *_latest), G_REPLAY
(drift vs the archived checkpoints at their own step), G_MOMENT (the
reconstructed v vs the LIVE optimizer's exp_avg_sq at the read states,
e240's formula), G_STEPS (per-step dnorm + applied_norm_live vs e254's
committed journal — the recursion certified against the committed product
at every step), G_BASIS (the committed-Gram basis re-derivation vs e234's
committed spectrum/knee/coverage), G_GRAM (the regenerated first-half
Gram vs the committed gram_scaled, fp16 cache tier), G_VSPAN (this cell's
span-mass reads vs e254's COMMITTED journal span_masses/span_masses2 at
the read states — the v-object reuse gate, TWO TIERS: the PRIMARY v
flavor must be BIT-EXACT (tol 1e-12; same fp64 recursion over the same
bit-exact replay), the v2 raw flavor compares at e254's fp32 SNAPSHOT
tier (tol 1e-6; e254 stored its raw-flavor snapshots as fp32, this cell
keeps fp64 — an instrument tier, not the reuse claim)), G_ROWS (the
y-side rows + the
recomputed span join == e256's committed join: n 216, |d rho| <= 1e-12).

PROVENANCE (extend-don't-repeat): builds on T231/e254 (the composition:
the denominator is the thermal channel's carrier candidate; THIS cell is
its registered sharper probe-side prediction), T230/e256 (the original
coupling: span-residency -> thermal share, rho -0.148 pooled; the join-3
row construction replicated verbatim), T218/e238 (the per-probe thermal
residuals — the y-side), T216/e239 (the 54 committed t=0 supports),
T225/e240 + e254's module-imported machinery (the fp64 v-reconstruction;
the certified replay; the supports bank), T217/e234 (the committed
wash-span bases + the certified gradient journal), T198/e226 (the support
bank + cache conventions), e182c/e182c2 (the w1/w2 archives + certified
draw streams). NEW: the first PROBE-LEVEL read of the optimizer's second
moment (the per-support quadratic form — nobody has measured the
denominator's load on each probe's own direction); the head-to-head
carrier comparison (v-overlap vs span-residency on identical rows); the
random-direction base-rate band for the quadratic form.

Envelope: desk-only CPU (e248 owns the GPU), progressive PARTIAL metrics +
journal after every phase, load checks per phase, scratch memmaps
(supports 13.4 GB fp16 + per-wash grads ~10 GB fp16) deleted on DONE /
kept on failure. No NOTES/THINKING/QUEUE/STATE edits (dispatch). Smoke
via E257_SMOKE=1 (12 steps, w1 only, read state {10}, 11 probes; nothing
adjudicated or gated).

Run:  cd lab && python e257_support_v_overlap.py     (E257_SMOKE=1 for
      the shakedown; nothing adjudicated or gated in smoke)
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# import e254 FIRST: it imports e240, which sets the offline/env/thread
# defaults before numpy/torch load (its module-level block is configuration
# only; both mains are __main__-guarded).
import e254_vspan_overlap as e254                # noqa: E402 — the machinery
import e240_moment_archive as e240               # noqa: E402 — the machinery

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

import common                                   # noqa: E402
from common import now_iso, run_dir, save_json  # noqa: E402

e1 = e240.e1                                    # e182c, via e240
e226 = e240.e226                                # via e240

import torch.nn.functional as F                 # noqa: E402
from scipy.stats import spearmanr               # noqa: E402

import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import textwrap                                  # noqa: E402

try:
    import psutil                                # noqa: E402
except ImportError:
    psutil = None

SMOKE = os.environ.get("E257_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e257_smoke" if SMOKE else "e257"

T0 = time.time()
_log_t0 = time.time()
def log(m: str) -> None:
    print(f"[{time.time() - _log_t0:8.1f}s] {m}", flush=True)

# ------------------------------------------------------------ the frozen cell
WASHES = ("w1",) if SMOKE else ("w1", "w2")     # the y-side's washes (e238)
STEPS = 12 if SMOKE else 80
HALF = STEPS // 2                               # the split-half boundary
KNEE_MAX_I = 5 if SMOKE else 20                 # e234's knee search head
K_CAP = 2 if SMOKE else 20                      # e234's cap
K_LADDER = (1, 2) if SMOKE else (1, 2, 5, 10, 20)
READ_STATES = {"w1": (10,) if SMOKE else (50, 80),
               "w2": (50, 80)}
CHUNK = e240.CHUNK                              # 2**22 (e226/e240 chunk)
CACHE_SCALE = e226.CACHE_SCALE                  # 2**14
N_NULL = 16 if SMOKE else 64                    # random-direction draws
NULL_SEED_BASE = 2_570_000                      # disclosed
NULL_SEED_STRIDE = 1_000_003                    # disclosed
NULL_CHUNK_STRIDE = 7919                        # disclosed
CHUNK_NULL = 1 << 21                            # 2^21 coords/null chunk
RAM_FLOOR_GB = 16.0                             # e226's cache floor
RAM_FLOOR_GRADS_GB = 12.0                       # e254's grads-memmap floor
RAM_WAIT_GRADS_S = 120.0                        # e254's politeness cap
LR, B1, B2, WD, EPS_ADAM, CLIP = (e240.LR, e240.B1, e240.B2,
                                  e240.WD, e240.EPS_ADAM, e240.CLIP)
SCRATCH = common.REPO / "scratch" / "e257_cache"

# committed records (load-only)
E182C_M = common.REPO / "runs" / "e182c" / "metrics.json"
E182C2_M = common.REPO / "runs" / "e182c2" / "metrics.json"
E216_M = common.REPO / "runs" / "e216" / "metrics.json"
E226_M = common.REPO / "runs" / "e226" / "metrics.json"
E234_M = common.REPO / "runs" / "e234" / "metrics.json"
E234_J = common.REPO / "runs" / "e234" / "journal.json"
E238_M = common.REPO / "runs" / "e238" / "metrics.json"
E240_M = common.REPO / "runs" / "e240" / "metrics.json"
E254_J = common.REPO / "runs" / "e254" / "journal.json"
E256_J = common.REPO / "runs" / "e256" / "journal.json"
E256_M = common.REPO / "runs" / "e256" / "metrics.json"

REGISTERED = {
    "bars_verbatim": {
        "V-OVERLAP-WINS": "the support-v-overlap join's |rho| exceeds the "
        "span-residency join's by >= 0.05 pooled (both computed here on "
        "identical probes) — the denominator load on the probe's OWN "
        "direction is the better carrier; the composition's probe-level "
        "evidence landed",
        "SPAN-SPECIFIC": "the v-overlap join does NOT beat the span join "
        "(within 0.05) — the carrier is the span basis specifically; the "
        "composition bounded",
        "MIXED": "the tables verbatim, both joins, per wash",
    },
    "question_verbatim": "does a probe's SUPPORT-V-OVERLAP (its support's "
    "share of v's mass — the direct denominator load on its own direction) "
    "predict its thermal share BETTER than the span-residency did (rho "
    "-0.148)? If yes: the composition's carrier claim gains its probe-level "
    "evidence; if no: the coupling's carrier is the span-basis specifically, "
    "not v-load generally.",
    "adjudication_rule_frozen": {
        "delta_pool": "|rho_v_overlap pooled 216 rows| - |rho_span pooled "
        "the same 216 rows| (both computed here on identical probes)",
        "delta_w": "the same, per wash (the identical 108-row subsets)",
        "discordant": "(delta_pool >= 0.05 AND min(delta_w) <= -0.05) OR "
        "(delta_pool < 0.05 AND min(delta_w) >= 0.05)",
        "V-OVERLAP-WINS": "delta_pool >= 0.05 AND NOT discordant",
        "SPAN-SPECIFIC": "delta_pool < 0.05 AND NOT discordant",
        "MIXED": "discordant",
        "note": "a wash hard-contradicts a pooled win only at the bar's own "
        "0.05 margin in the opposite direction; ties are not contradictions",
    },
    "registration": "bars + question + adjudication rule frozen VERBATIM "
    "from the dispatch brief BEFORE this script or any compute; the script "
    "was committed (git) before the first full run; adjudicate against "
    "exactly this; no bar shopping",
}

deviations: list[str] = [
    "v is NOT committed as a vector anywhere (e254 committed its span "
    "masses only) — it is RECONSTRUCTED here by e240's fp64 recursion over "
    "the certified wash replays (module-import, e254's replay loop "
    "statements verbatim, same order/chunking), and certified four ways: "
    "G_CE/G_DRAWS/G_REPLAY (bit-exact replay vs e234's journal + the "
    "archived draw/checkpoint streams), G_MOMENT (v vs the live optimizer "
    "at the read states), G_STEPS (per-step dnorm/applied vs e254's "
    "committed journal), and G_VSPAN (my span masses == e254's committed "
    "span_masses/span_masses2 at the read states — the v-object reuse "
    "gate).",
    "The supports are t=0-fixed (e226/e239's committed bank, rebuilt by "
    "e240.build_supports and certified by FD + the e226 w1:s0 alignment, "
    "dp <= 0.005); their fp16 cache floor (~5e-4 per cos) rides every "
    "quadratic form (disclosed). The supports memmap must SURVIVE into the "
    "replay phase (unlike e254, which released it — this cell reads it); "
    "deleted on DONE.",
    "w3 is NOT replayed: e238's committed residual cells (the frozen "
    "y-side) exist only for {w1, w2} x {50, 80}; the join scope IS the "
    "y-side's scope, and the span join is recomputed on the IDENTICAL 216 "
    "rows (G_ROWS certifies the recomputation against e256's committed "
    "rho -0.14793697753319948).",
    "torch threads 8 (e182c's bit-exact CPU replay convention, VERBATIM "
    "from e240/e254 — the w1 product is a threads-8 product); the "
    "dispatch's threads-4 clause is honored on the numpy/BLAS side "
    "(OMP/MKL 4, exactly as e254's G_ENV under the same desk-only "
    "dispatch regime with e248 owning the GPU).",
    "The x-side is STATE-LEVEL (v at the residual cell's own state) while "
    "e256's x-side was WASH-LEVEL standing geometry (t=0 residency, "
    "repeated across the wash's states) — this is the registered sharpener "
    "(the denominator load AT the state), disclosed as a structural "
    "difference between the two joins' x-sides.",
    "The coarse battery-level co-report uses the wash-level battery MEDIAN "
    "qov (pooled over the wash's 2 read states), repeated across the fit "
    "cell's states — mirroring e256's coarse-read repetition disclosure.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "INSTRUMENT-TIER FIX (envelope-only, no registered quantity touched; "
    "the e254 RELAUNCH-DISCLOSURE precedent): the first full session ran "
    "w1 with G_VSPAN's raw-flavor (v2) comparison at the primary's 1e-12 "
    "tier and it FAILED at 2.34e-8 — diagnosed to e254's fp32 v2 "
    "SNAPSHOT quantization (e254 stored snaps2 as fp32; this cell keeps "
    "fp64), while the PRIMARY v flavor was BIT-EXACT (max |d| exactly "
    "0.0). The gate was re-tiered BEFORE any join was computed (the run "
    "halted at the gate): primary tol 1e-12, v2 tol 1e-6 (the fp32 "
    "snapshot tier, disclosed above). No bar, no join, no x/y value was "
    "touched; the replays were NOT re-run (journal-resume path, every "
    "number re-certified from the journal).",
    "Smoke mode (E257_SMOKE=1): 12 steps, w1 only, read state {10}, 11 "
    "probes, e254.HALF patched to 6 for the cache-tier helpers; nothing "
    "adjudicated or gated (SMOKE stamp).",
]


# ------------------------------------------------------------ envelope

def cpu_load_check(tag: str) -> dict:
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    rec = {"tag": tag,
           "cpu_percent": psutil.cpu_percent(interval=0.5),
           "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1)}
    log(f"  [load] {tag}: cpu {rec['cpu_percent']}% "
        f"ram_avail {rec['ram_avail_gb']} GB")
    return rec


def ram_wait(floor_gb: float, tag: str, max_wait_s: float = 600.0) -> dict:
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    t0 = time.time()
    while psutil.virtual_memory().available / 2**30 < floor_gb:
        if time.time() - t0 > max_wait_s:
            log(f"  [ram-wait] {tag}: floor {floor_gb} GB NOT reached in "
                f"{max_wait_s}s — proceeding anyway (disclosed)")
            return {"tag": tag, "waited_s": round(time.time() - t0, 1),
                    "floor_gb": floor_gb, "cleared": False}
        time.sleep(20)
    return {"tag": tag, "waited_s": round(time.time() - t0, 1),
            "floor_gb": floor_gb, "cleared": True}


def save_journal(rd: Path, journal: dict):
    (rd / "journal.json").write_text(
        json.dumps(journal, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------ the replay

def replay_wash(net0, train_ids, sup, journal, w: str, e234_wj, read_states):
    """One wash replayed with e254's VERBATIM arithmetic (its chunk-loop
    statements, same order — the m_/v_/v2_ recursions are bit-identical by
    construction), extended with: (a) the first-half RAW pre-clip gradients
    streamed to an fp16 memmap (e234's cache convention — the G_VSPAN/G_GRAM
    substrate); (b) fp64 v/v2 snapshots at the read states; (c) the
    live-optimizer moment certification at the read states."""
    S = STEPS
    N = sum(p.numel() for p in net0.parameters())
    wj = journal.setdefault(w, {"steps": {}})

    SCRATCH.mkdir(parents=True, exist_ok=True)
    gm_path = SCRATCH / f"grads_{w}.npy"
    gm = np.lib.format.open_memmap(gm_path, mode="w+", dtype=np.float16,
                                   shape=(HALF, N))

    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(B1, B2),
                            weight_decay=WD)
    gen = torch.Generator().manual_seed(e240.W_SEED[w])
    hi = train_ids.shape[0] - e1.SEQ - 1

    m_ = torch.zeros(N, dtype=torch.float64)        # e240's m_ (clipped)
    v_ = torch.zeros(N, dtype=torch.float64)        # PRIMARY v (clipped)
    v2_ = torch.zeros(N, dtype=torch.float64)       # RAW flavor
    snaps, snaps2, moment_certs = {}, {}, {}
    t_start = time.time()

    for step in range(1, S + 1):
        off = torch.randint(hi, (e1.BATCH,), generator=gen)
        x = torch.stack([train_ids[o: o + e1.SEQ] for o in off])
        y = torch.stack([train_ids[o + 1: o + 1 + e1.SEQ] for o in off])
        logits = net(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        ce = float(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_raw = e226.flat_grad(net)                     # fp32 flat
        nrm = float(g_raw.norm(dtype=torch.float64).item())
        if not (nrm > 0 and math.isfinite(nrm)):
            raise RuntimeError(f"{w} s{step}: degenerate gradient")
        tn = torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        tnf = float(tn)
        clip_fac = min(1.0, 1.0 / (tnf + 1e-6))         # torch's formula

        pb = [p.detach().clone() for p in net.parameters()]
        theta_b = torch.cat([t.reshape(-1) for t in pb])

        b1c = 1.0 - B1 ** step
        b2c = 1.0 - B2 ** step
        d2 = arec = 0.0
        inv = 1.0 / nrm
        for a in range(0, N, CHUNK):
            b = min(a + CHUNK, N)
            raw64 = g_raw[a:b].to(torch.float64)
            gc64 = raw64 * clip_fac
            # e240/e254's VERBATIM recursion statements (same order/chunking)
            m_[a:b].mul_(B1).add_(gc64, alpha=1.0 - B1)
            v_[a:b].mul_(B2).addcmul_(gc64, gc64, value=1.0 - B2)
            v2_[a:b].mul_(B2).addcmul_(raw64, raw64, value=1.0 - B2)
            d = (m_[a:b] / b1c) / ((v_[a:b] / b2c).sqrt() + EPS_ADAM)
            d2 += float(d.dot(d))
            arec += float((WD * theta_b[a:b].to(torch.float64) + d
                           ).pow(2).sum())
            if step <= HALF:   # cache the RAW pre-clip row (e234's conv.)
                row = ((raw64 * inv) * CACHE_SCALE).to(torch.float16)
                gm[step - 1, a:b] = row.numpy()
                del row
            del d, raw64, gc64
        opt.step()
        live2 = 0.0
        with torch.no_grad():
            for p, pbb in zip(net.parameters(), pb):
                live2 += float((p - pbb).pow(2).sum(dtype=torch.float64))
        del pb, theta_b, g_raw

        applied_rec = LR * math.sqrt(max(arec, 0.0))
        applied_live = math.sqrt(max(live2, 0.0))
        wj["steps"][str(step)] = {
            "ce": ce, "gnorm_raw": nrm, "tn_torch": tnf,
            "clip_fac": clip_fac, "applied_norm_rec": applied_rec,
            "applied_norm_live": applied_live,
            "dnorm": math.sqrt(max(d2, 0.0)),
        }

        # ---- G_REPLAY: drift vs the archived checkpoint at its own step
        if not SMOKE and step in e240.W_ARCH[w]:
            wj.setdefault("replay_cert", {})[str(step)] = \
                e240.drift_vs_archived(net, e240.W_ARCH[w][step])

        # ---- read states: G_MOMENT + the fp64 v/v2 snapshots
        if not SMOKE and step in read_states:
            ma = torch.cat([opt.state[p]["exp_avg"].reshape(-1)
                            for p in net.parameters()])
            va = torch.cat([opt.state[p]["exp_avg_sq"].reshape(-1)
                            for p in net.parameters()])
            rel_m = float((m_ - ma.to(torch.float64)).abs().max()
                          / ma.abs().max().clamp(min=1e-30))
            rel_v = float((v_ - va.to(torch.float64)).abs().max()
                          / va.abs().max().clamp(min=1e-30))
            del ma, va
            moment_certs[str(step)] = {"rel_m_max": rel_m, "rel_v_max": rel_v}
            wj.setdefault("moment_certs", {})[str(step)] = \
                moment_certs[str(step)]
            snaps[step] = v_.clone()
            snaps2[step] = v2_.clone()

        if step % 10 == 0 or step == S:
            log(f"  [{w} replay] s{step:3d}/{S} CE {ce:.4f} |g| {nrm:.2f} "
                f"clip {clip_fac:.4f} applied {applied_live:.4f} "
                f"({time.time() - t_start:.0f}s)")
            journal[w] = wj
            save_journal(RD, journal)
    gm.flush()

    # ---- G_DRAWS: bit-exact generator state vs the archived *_latest
    gen_ok = None
    if not SMOKE:
        archived = torch.load(e240.W_LATEST[w], map_location=CPU,
                              weights_only=False)["gen"]
        gen_ok = bool(torch.equal(gen.get_state(), archived))
        wj["gen_ok"] = gen_ok

    # ---- G_CE: bit-exact vs e234's journaled CE + gnorms
    ce_cert = gnorm_cert = None
    if e234_wj is not None:
        ce_cert = max(abs(wj["steps"][str(t)]["ce"] - e234_wj["ces"][t - 1])
                      for t in range(1, S + 1))
        gnorm_cert = max(abs(wj["steps"][str(t)]["gnorm_raw"]
                             - e234_wj["gnorms"][t - 1])
                         for t in range(1, S + 1))

    journal[w] = wj
    save_journal(RD, journal)
    log(f"{w}: replay done ({time.time() - t_start:.0f}s); gen_ok {gen_ok}; "
        f"dCE {ce_cert}; dgnorm {gnorm_cert}")
    return {"gen_ok": gen_ok, "ce_cert": ce_cert, "gnorm_cert": gnorm_cert,
            "replay_cert": wj.get("replay_cert", {}),
            "moment_certs": moment_certs, "snaps": snaps, "snaps2": snaps2,
            "gm_path": gm_path, "gm": gm}


# ------------------------------------------------- post-replay reads (per wash)

def qov_reads(gm, snaps, snaps2, sup_mm, row_norms, seed_tag: int):
    """THE X-SIDE + the base rates, per read state: q_i = sum_p mm[i,p]^2
    * v[p] (the quadratic form on the CACHED support rows); the unit-support
    form divides by the cached row norms squared; the null draws are exact-
    normalized gaussian unit directions (seed_tag separates the washes)."""
    N = gm.shape[1]
    n_p = sup_mm.shape[0]
    out = {}
    rn2 = np.asarray(row_norms, dtype=np.float64) ** 2
    for st in sorted(snaps):
        v_st, v2_st = snaps[st], snaps2[st]
        q = torch.zeros(n_p, dtype=torch.float64)
        q2 = torch.zeros(n_p, dtype=torch.float64)
        v1_sum = v2_1sum = v1_sq = 0.0
        num = torch.zeros(N_NULL, dtype=torch.float64)
        den = torch.zeros(N_NULL, dtype=torch.float64)
        for a in range(0, N, CHUNK):
            b = min(a + CHUNK, N)
            Sb = torch.from_numpy(
                np.ascontiguousarray(sup_mm[:, a:b])).to(torch.float64)
            sq = Sb * Sb
            q += sq @ v_st[a:b]
            q2 += sq @ v2_st[a:b]
            v1_sum += float(v_st[a:b].sum())
            v2_1sum += float(v2_st[a:b].sum())
            v1_sq += float((v_st[a:b] * v_st[a:b]).sum())
            # the random-direction null (chunk-wise, disclosed seeds)
            ci = a // CHUNK_NULL
            g = torch.Generator().manual_seed(
                NULL_SEED_BASE + NULL_SEED_STRIDE * st
                + NULL_CHUNK_STRIDE * (ci + 512 * seed_tag))
            Z = torch.randn(N_NULL, b - a, generator=g,
                            dtype=torch.float32).to(torch.float64)
            Zsq = Z * Z
            num += Zsq @ v_st[a:b]
            den += Zsq.sum(dim=1)
            del Sb, sq, Z, Zsq
        qv = (q / torch.from_numpy(rn2)).numpy()
        qv2 = (q2 / torch.from_numpy(rn2)).numpy()
        qnull = (num / den).numpy()
        out[st] = {
            "qov_unit": qv.tolist(),          # <s_hat, V s_hat> (PRIMARY)
            "qov2_unit": qv2.tolist(),        # raw-gradient flavor
            "null_draws": qnull.tolist(),     # random unit directions
            "base_rate_mean_v": v1_sum / N,   # Haar E for a unit direction
            "base_rate_mean_v2": v2_1sum / N,
            "v_l2_sq": v1_sq,
            "participation_ratio": (v1_sum * v1_sum) / v1_sq
            if v1_sq > 0 else float("nan"),
        }
        log(f"  [qov] state {st}: median qov {np.median(qv):.3e} "
            f"(base rate {v1_sum / N:.3e}, null median "
            f"{np.median(qnull):.3e})")
    return out


# ------------------------------------------------------------ main

def gen_verify(w: str, train_ids) -> bool | None:
    """draws-only generator re-verification (no model needed): reseed,
    replay STEPS randint draws, compare to the archived *_latest (e240's
    convention — the resume path's G_DRAWS)."""
    if SMOKE:
        return None
    gen = torch.Generator().manual_seed(e240.W_SEED[w])
    hi = train_ids.shape[0] - e1.SEQ - 1
    for _ in range(STEPS):
        torch.randint(hi, (e1.BATCH,), generator=gen)
    archived = torch.load(e240.W_LATEST[w], map_location=CPU,
                          weights_only=False)["gen"]
    return bool(torch.equal(gen.get_state(), archived))


RD = None

def main() -> int:
    global RD
    rd = run_dir(NAME)
    RD = rd
    jp = rd / "journal.json"
    log(f"E257 — THE SUPPORT-V-OVERLAP READ (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e257_support_v_overlap",
        "phase": "T231's registered sharper prediction: the per-probe "
                 "support-v-overlap <s_i, V_s s_i>/||s_i||^2 (v = e240's "
                 "fp64 reconstruction at e238's residual states) joined to "
                 "the per-probe thermal deviation |resid_z|, head-to-head "
                 "against e256's span-residency join on identical rows",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered_prediction": REGISTERED,
        "question": REGISTERED["question_verbatim"],
        "builds_on": [
            "T231 / e254 (the composition: the denominator owns the span; "
            "the carrier candidate; THIS cell is its registered sharper "
            "probe-side prediction)",
            "T230 / e256 (the original coupling: span-residency -> thermal "
            "share, rho -0.148 pooled; the join-3 row construction "
            "replicated verbatim on identical rows)",
            "T218 / e238 (the per-probe thermal residuals |resid_z| — the "
            "frozen y-side)",
            "T216 / e239 (the 54 committed t=0 supports)",
            "T225 / e240 + e254 (the fp64 v-reconstruction machinery, "
            "module-imported; the certified replay; the supports bank)",
            "T217 / e234 (the committed wash-span bases + the certified "
            "gradient journal; the cache conventions)",
            "T198 / e226 (the support bank + cache conventions)",
            "e182c / e182c2 (the w1/w2 archives + certified draw streams)",
        ],
        "whats_new": [
            "the first PROBE-LEVEL read of the optimizer's second moment: "
            "the per-support quadratic form <s_i, V_s s_i>/||s_i||^2 (the "
            "denominator's load on each probe's OWN direction, full form "
            "over all 124M coordinates)",
            "the head-to-head carrier comparison: the v-overlap join vs "
            "the span-residency join on IDENTICAL 216 rows (the registered "
            "0.05 bar)",
            "the random-direction base-rate band for the quadratic form "
            "(exact-normalized gaussian unit draws) + v's participation "
            "ratio — the arithmetic-pre-announcement disclosure",
            "the v-object reuse gate (G_VSPAN): this cell's span masses == "
            "e254's committed journal values at the read states",
        ],
        "smoke": SMOKE,
    }

    def write_metrics(status: str):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    journal: dict = {}
    if jp.exists():
        try:
            journal = json.loads(jp.read_text(encoding="utf-8"))
            log(f"journal restored: {list(journal)}")
        except Exception as e:                              # noqa: BLE001
            log(f"journal unreadable ({e}); starting fresh")
            journal = {}

    load_checks: list[dict] = [cpu_load_check("launch")]
    metrics["load_checks"] = load_checks
    ram_waits: list[dict] = []

    # ------------------------------------------------ P0 the committed records
    for p in (E182C_M, E182C2_M, E216_M, E226_M, E234_M, E234_J, E238_M,
              E240_M, E254_J, E256_J, E256_M):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e226m = json.loads(E226_M.read_text(encoding="utf-8"))
    e234m = json.loads(E234_M.read_text(encoding="utf-8"))
    e234j = json.loads(E234_J.read_text(encoding="utf-8"))
    e238m = json.loads(E238_M.read_text(encoding="utf-8"))
    e216m = json.loads(E216_M.read_text(encoding="utf-8"))
    p1m = json.loads(E182C_M.read_text(encoding="utf-8"))
    c2m = json.loads(E182C2_M.read_text(encoding="utf-8"))
    j254 = json.loads(E254_J.read_text(encoding="utf-8"))
    j256 = json.loads(E256_J.read_text(encoding="utf-8"))
    m256 = json.loads(E256_M.read_text(encoding="utf-8"))
    metrics["provenance_records"] = {
        "v_machinery": "lab/e240_moment_archive.py + lab/e254_vspan_"
                       "overlap.py (module-import; the replay loop "
                       "statements verbatim)",
        "v_object_reuse_targets": {"e254_journal": str(E254_J),
                                   "fields": "span_masses / span_masses2 "
                                             "at the read states"},
        "span_bases+gram": str(E234_J),
        "e234_metrics": str(E234_M),
        "supports+alignments": str(E226_M),
        "y_side_thermal": str(E238_M),
        "span_join_census+committed": {"journal": str(E256_J),
                                       "metrics": str(E256_M)},
        "washes": "runs/e182c (w1), runs/e182c2 (w2)",
    }

    # ------------------------------------------------ P1 organism + batteries
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = {**org_meta, "torch_threads": 8}
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"]
    metrics["size_gate"] = G_SIZE

    bats, train_ids, G_CORPUS = e240.build_batteries(net0, tok)
    metrics["gates"] = {"G_SIZE": G_SIZE, "G_CORPUS": G_CORPUS,
                        "G_ENV": {
                            "desk_only": True, "gpu_used": False,
                            "torch_threads": 8,
                            "torch_threads_note": "e182c's bit-exact CPU "
                                "replay convention, VERBATIM from e240/e254",
                            "blas_threads": 4,
                            "blas_note": "OMP/MKL 4 — the dispatch's "
                                "threads-4 clause (e254's G_ENV precedent)",
                            "pass": True}}
    log("G_CORPUS: " + ("PASS" if G_CORPUS["pass"] else "FAIL"))
    assert G_CORPUS["pass"] or SMOKE
    write_metrics("PARTIAL: organism + corpus certified")

    w1_rec = {s["step"]: s for s in p1m["states"]}
    w1_tmpl_rec = {s["step"]: s for s in c2m["part1_template"]["states"]}
    e216_rows = {r["fact"]: r for r in e216m["residual"]["table"]}
    assert len(e216_rows) == 54

    def _probes(state_rec, batt):
        return {f: v["p"] for f, v in state_rec[batt]["probes"].items()}
    G_BATT = {}
    for b, bl in bats.items():
        mine_p = {r["fact"]: r["p"] for r in bl}
        ref1 = _probes(w1_rec[0], b) if b != "tmpl" \
            else _probes(w1_tmpl_rec[0], "tmpl")
        G_BATT[b] = {"n": len(bl),
                     "set_equal_committed": bool(set(mine_p) == set(ref1)),
                     "max_dp_w1": max(abs(mine_p[f] - v)
                                      for f, v in ref1.items())}
    G_BATT["pass"] = bool(all(G_BATT[b]["set_equal_committed"]
                              and G_BATT[b]["max_dp_w1"] <= 0.010
                              for b in bats)) or SMOKE
    metrics["gates"]["G_BATT"] = G_BATT
    log("G_BATT: " + ("PASS" if G_BATT["pass"] else "FAIL"))
    if not G_BATT["pass"]:
        write_metrics("PARTIAL: G_BATT FAILED — halted")
        return 1

    probes54: list[dict] = []
    for b in ("fact", "ctrl", "near", "tmpl"):
        for r in bats[b]:
            probes54.append({**r, "battery": b,
                             "family6": e216_rows[r["fact"]]["family6"]})
    assert len(probes54) == 54
    N_PARAMS = sum(p.numel() for p in net0.parameters())

    # G_ORDER: my 54 == e226's committed 54 (facts, batteries, family6,
    # order) — the join's row alignment (e256's G_ORDER convention)
    names = [p["fact"] for p in e226m["probes"]]
    battery_of = {p["fact"]: p["battery"] for p in e226m["probes"]}
    fam_of = {p["fact"]: p["family6"] for p in e226m["probes"]}
    if not SMOKE:
        G_ORDER = {
            "rule": "my rebuilt 54 == e226's committed 54 (facts, order, "
                    "batteries, family6)",
            "facts_equal": bool([r["fact"] for r in probes54] == names),
            "batteries_equal": bool(all(r["battery"] == battery_of[r["fact"]]
                                        for r in probes54)),
            "family6_equal": bool(all(r["family6"] == fam_of[r["fact"]]
                                      for r in probes54)),
            "n": len(probes54)}
        G_ORDER["pass"] = bool(G_ORDER["facts_equal"]
                               and G_ORDER["batteries_equal"]
                               and G_ORDER["family6_equal"])
    else:
        probes54 = [r for r in probes54 if r["fact"] in (
            e226.ANCHOR_G, e226.ANCHOR_I,
            "The gaming console made by Microsoft->Xbox",
            "The web browser made by Google->Chrome",
            "The tablet made by Apple->iPad",
            "The music store made by Apple->iTunes",
            "The game console made by Sony->PlayStation",
            "The social network founded by Mark Zuckerberg->Facebook",
            "France->Paris", "Massachusetts->Boston",
            "Georgia->Atlanta")]
        log(f"SMOKE: probe subset n={len(probes54)}")
        G_ORDER = {"pass": True, "smoke_subset_n": len(probes54)}
    metrics["gates"]["G_ORDER"] = G_ORDER
    metrics["probes"] = [{"i": i, "fact": r["fact"], "battery": r["battery"],
                          "family6": r["family6"]} for i, r in
                         enumerate(probes54)]
    log(f"G_ORDER: {'PASS' if G_ORDER['pass'] else 'FAIL'}")
    if not G_ORDER["pass"]:
        write_metrics("PARTIAL: G_ORDER FAILED — halted")
        return 1

    # ------------------------------------------------ P2 the supports bank
    load_checks.append(cpu_load_check("supports build"))
    ram_waits.append(ram_wait(RAM_FLOOR_GB, "supports memmap"))
    metrics["ram_waits"] = ram_waits
    e240.SCRATCH = SCRATCH           # this cell's scratch (e240's is free)
    sup = e240.build_supports(net0, probes54, len(probes54), N_PARAMS,
                              journal)
    G_SUPPORT = e240.certify_supports(net0, train_ids, probes54, sup, e226m)
    metrics["gates"]["G_SUPPORT"] = G_SUPPORT
    log(f"G_SUPPORT: fd {G_SUPPORT['fd_pass_n']}/{len(probes54)}; probe-0 "
        f"self-cos {G_SUPPORT['determinism_selfcos_probe0']:.6f}; max dp vs "
        f"e226 w1:s0 "
        f"{G_SUPPORT['max_dp_vs_e226_committed_w1s0_alignment']:.5f} -> "
        f"{'PASS' if G_SUPPORT['pass'] else 'FAIL'}")
    if not G_SUPPORT["pass"] and not SMOKE:
        write_metrics("PARTIAL: G_SUPPORT FAILED — halted before replays")
        return 1
    sup["probes54"] = probes54
    save_journal(rd, journal)
    write_metrics("PARTIAL: supports bank rebuilt + certified vs e226")

    # ------------------------------------------------ P3 the bases (committed)
    bases = {}
    for w in WASHES:
        Gs = np.array(e234j[f"gram_scaled_{w}"], dtype=np.float64)
        gn = np.array(e234j[w]["gnorms"], dtype=np.float64)
        Gn = e254.true_gram(Gs, gn)
        k, cA, evec, ev, meta = e254.basis_coeffs(Gn, HALF, KNEE_MAX_I,
                                                  K_CAP)
        bases[w] = {"Gn": Gn, "gnorms": gn, "kA": k, "evec": evec,
                    "evals": ev, "meta": meta,
                    "cov_med": e254.coverage_median(Gn, gn, cA, HALF)}
        log(f"basis {w}: knee k={k}, var_frac@k "
            f"{meta['var_frac_at_k']:.4f}, coverage median "
            f"{bases[w]['cov_med']:.5f}")

    G_BASIS = {"per_wash": {}, "tol_rel_spectrum": 1e-9,
               "tol_abs_coverage": 1e-9}
    gb_ok = not SMOKE
    for w in WASHES:
        if SMOKE:
            break
        committed = e234m["decomposition"][w]
        spec_c = np.array(committed["metaA"]["spectrum"], dtype=np.float64)
        spec_m = np.array(bases[w]["evals"][:len(spec_c)])
        rel = float(np.abs(spec_m - spec_c).max() / spec_c.max())
        cov_d = abs(bases[w]["cov_med"] - committed["coverage_A"]["median"])
        knee_ok = bases[w]["kA"] == committed["kA"]
        G_BASIS["per_wash"][w] = {
            "max_rel_spectrum_dev": rel, "knee_matches": knee_ok,
            "knee": bases[w]["kA"], "committed_knee": committed["kA"],
            "abs_coverage_dev": cov_d,
            "var_frac_at_k": bases[w]["meta"]["var_frac_at_k"],
            "committed_var_frac": committed["metaA"]["var_frac_at_k"]}
        gb_ok = gb_ok and rel <= 1e-9 and knee_ok and cov_d <= 1e-9
    G_BASIS["pass"] = bool(gb_ok) or SMOKE
    metrics["gates"]["G_BASIS"] = G_BASIS
    log("G_BASIS: " + ("PASS" if G_BASIS["pass"] else "FAIL"))
    write_metrics("PARTIAL: committed bases re-derived + certified")

    cA_ladders = {w: e254.ladder_from(bases[w]["evec"], bases[w]["evals"])
                  for w in WASHES}

    # ------------------------------------------------ P4 the replays + reads
    wash_outputs = {}
    for w in WASHES:
        # the resume path: a fully-read wash (reads_done in the journal,
        # from a prior killed session of THIS run) is not re-derived —
        # its certs are recomputed from the journal (e240's convention)
        if (not SMOKE and journal.get(w, {}).get("reads_done")
                and journal[w].get("qov_reads")
                and len(journal[w].get("steps", {})) == STEPS):
            log(f"{w}: reads already done in journal — skipping replay "
                f"(certs recomputed from the journal)")
            wj = journal[w]
            e234_wj = e234j.get(w)
            ce_cert = gnorm_cert = None
            if e234_wj is not None:
                ce_cert = max(abs(wj["steps"][str(t)]["ce"]
                                  - e234_wj["ces"][t - 1])
                              for t in range(1, STEPS + 1))
                gnorm_cert = max(abs(wj["steps"][str(t)]["gnorm_raw"]
                                     - e234_wj["gnorms"][t - 1])
                                 for t in range(1, STEPS + 1))
            wash_outputs[w] = {
                "gen_ok": gen_verify(w, train_ids),
                "ce_cert": ce_cert, "gnorm_cert": gnorm_cert,
                "replay_cert": wj.get("replay_cert", {}),
                "moment_certs": wj.get("moment_certs", {}),
                "resumed": True}
            # rehydrate the per-wash gate records from the journal
            for gname in ("G_STEPS", "G_GRAM", "G_VSPAN"):
                rec = wj.get("gate_records", {}).get(gname)
                if gname == "G_VSPAN":
                    # recomputed from the journal's own primary-flavor
                    # rows (bit-exactness of v — the reuse claim); the
                    # fp32-tier v2 co-report is not re-derived on resume
                    vrows = wj.get("vspan_rows", [])
                    devs_p = [abs(r["mine"] - r["committed"])
                              for r in vrows
                              if r.get("flavor", "v_primary")
                              == "v_primary"]
                    rec = {"n_values_primary": len(devs_p),
                           "max_abs_dev_primary": max(devs_p)
                           if devs_p else None,
                           "tol_primary": 1e-12,
                           "v2_on_resume": "not re-derived (verified in "
                                           "the original session's "
                                           "record, fp32 tier)",
                           "pass": bool(devs_p and max(devs_p) <= 1e-12)}
                    metrics["gates"].setdefault(gname, {})[w] = rec
                    wj.setdefault("gate_records", {})[gname] = rec
                    continue
                if rec is not None:
                    metrics["gates"].setdefault(gname, {})[w] = rec
                elif gname == "G_STEPS":   # recomputable from dicts alone
                    ddn = max(abs(wj["steps"][str(t)]["dnorm"]
                                  - j254[w]["steps"][str(t)]["dnorm"])
                              for t in range(1, STEPS + 1))
                    dap = max(abs(wj["steps"][str(t)]["applied_norm_live"]
                                  - j254[w]["steps"][str(t)]
                                  ["applied_norm_live"])
                              for t in range(1, STEPS + 1))
                    rel_ddn = ddn / max(j254[w]["steps"][str(t)]["dnorm"]
                                        for t in range(1, STEPS + 1))
                    rec = {"max_abs_d_dnorm": ddn,
                           "max_abs_d_applied_live": dap,
                           "max_rel_d_dnorm": rel_ddn, "tol_rel": 1e-9,
                           "pass": bool(rel_ddn <= 1e-9 and dap <= 1e-9)}
                    metrics["gates"].setdefault(gname, {})[w] = rec
                    wj.setdefault("gate_records", {})[gname] = rec
            save_journal(rd, journal)
            continue
        load_checks.append(cpu_load_check(f"replay {w}"))
        ram_waits.append(ram_wait(RAM_FLOOR_GRADS_GB, f"grads memmap {w}",
                                  max_wait_s=RAM_WAIT_GRADS_S))
        metrics["ram_waits"] = ram_waits
        read_states = READ_STATES[w]
        e234_wj = e234j.get(w) if not SMOKE else None
        out = replay_wash(net0, train_ids, sup, journal, w, e234_wj,
                          read_states)
        out["resumed"] = False
        wash_outputs[w] = out
        write_metrics(f"PARTIAL: replay {w} done (gen_ok {out['gen_ok']})")
        gm = out["gm"]
        gnorms_mine = np.array([journal[w]["steps"][str(t)]["gnorm_raw"]
                                for t in range(1, HALF + 1)])

        # ---- G_STEPS: per-step dnorm + applied_norm_live vs e254's journal
        g_steps = None
        if not SMOKE and w in j254:
            ddn = max(abs(journal[w]["steps"][str(t)]["dnorm"]
                          - j254[w]["steps"][str(t)]["dnorm"])
                      for t in range(1, STEPS + 1))
            dap = max(abs(journal[w]["steps"][str(t)]["applied_norm_live"]
                          - j254[w]["steps"][str(t)]["applied_norm_live"])
                      for t in range(1, STEPS + 1))
            rel_ddn = ddn / max(j254[w]["steps"][str(t)]["dnorm"]
                                for t in range(1, STEPS + 1))
            g_steps = {"max_abs_d_dnorm": ddn, "max_abs_d_applied_live": dap,
                       "max_rel_d_dnorm": rel_ddn, "tol_rel": 1e-9}
            g_steps["pass"] = bool(rel_ddn <= 1e-9 and dap <= 1e-9)
            metrics["gates"].setdefault("G_STEPS", {})[w] = g_steps
            journal[w].setdefault("gate_records", {})["G_STEPS"] = g_steps
            log(f"G_STEPS {w}: rel ddnorm {rel_ddn:.2e}, d applied "
                f"{dap:.2e} -> {'PASS' if g_steps['pass'] else 'FAIL'}")

        # ---- G_GRAM: the regenerated Gram vs the committed one
        G_regen = e254.gram_from_memmap(gm)
        k_r, _cAr, _evr, evr, _mr = e254.basis_coeffs(
            e254.true_gram(G_regen, gnorms_mine), HALF, KNEE_MAX_I, K_CAP)
        if not SMOKE:
            Gs_c = np.array(e234j[f"gram_scaled_{w}"], dtype=np.float64)
            scale = float(CACHE_SCALE ** 2)
            dcos = float(np.abs(G_regen / scale
                                - Gs_c[:HALF, :HALF] / scale).max())
            n_cmp = min(20, len(evr), len(bases[w]["evals"]))
            eig_rel = float(np.abs(
                np.array(evr[:n_cmp])
                - np.array(bases[w]["evals"][:n_cmp])).max()
                / bases[w]["evals"][0])
            ggram = {"max_abs_dcos": dcos, "tol_dcos": 2e-3,
                     "regen_knee": int(k_r),
                     "committed_knee": int(bases[w]["kA"]),
                     "max_rel_eig_dev_top20": eig_rel, "tol_eig": 0.01}
            ggram["pass"] = bool(dcos <= 2e-3 and k_r == bases[w]["kA"]
                                 and eig_rel <= 0.01)
            metrics["gates"].setdefault("G_GRAM", {})[w] = ggram
            journal[w].setdefault("gate_records", {})["G_GRAM"] = ggram
            log(f"G_GRAM {w}: dcos {dcos:.2e}, knee {k_r} vs "
                f"{bases[w]['kA']}, eig rel {eig_rel:.2e} -> "
                f"{'PASS' if ggram['pass'] else 'FAIL'}")

        # ---- THE X-SIDE: the per-probe quadratic forms + base rates
        reads = qov_reads(gm, out["snaps"], out["snaps2"], sup["mm"],
                          sup["row_norms"], seed_tag=WASHES.index(w))
        journal[w]["qov_reads"] = {str(st): rec for st, rec in reads.items()}

        # ---- G_VSPAN: my span masses vs e254's COMMITTED journal values
        # (two tiers, fixed before this cell's adjudication and disclosed:
        # the PRIMARY v flavor must be BIT-EXACT — same fp64 recursion,
        # same replay (G_STEPS dev 0.0); the v2 raw flavor is compared at
        # e254's fp32 SNAPSHOT tier — e254 stored its raw-flavor snapshots
        # as fp32, this cell keeps fp64, so the tier is the quantization)
        if not SMOKE:
            lad = cA_ladders[w]
            devs_p, devs_v2 = [], []
            rows_vspan = []
            for st in sorted(out["snaps"]):
                vnorm2 = float(out["snaps"][st].double().pow(2).sum())
                gdotv = e254.dots_vs_cache(gm, gnorms_mine, out["snaps"][st])
                masses = e254.span_masses(gdotv, vnorm2, lad)
                gd2r = e254.dots_vs_cache(gm, gnorms_mine,
                                          out["snaps2"][st])
                v2norm2 = float(out["snaps2"][st].double().pow(2).sum())
                masses2 = e254.span_masses(gd2r, v2norm2, lad)
                del gdotv, gd2r
                com = j254[w]["span_masses"].get(str(st), {})
                com2 = j254[w].get("span_masses2", {}).get(str(st), {})
                for k in K_LADDER:
                    if str(k) in com:
                        devs_p.append(abs(masses[k] - com[str(k)]))
                        rows_vspan.append({"state": st, "k": k,
                                           "flavor": "v_primary",
                                           "mine": masses[k],
                                           "committed": com[str(k)]})
                    if str(k) in com2:
                        devs_v2.append(abs(masses2[k] - com2[str(k)]))
            gv = {"n_values_primary": len(devs_p),
                  "max_abs_dev_primary": max(devs_p) if devs_p else None,
                  "tol_primary": 1e-12,
                  "n_values_v2_raw": len(devs_v2),
                  "max_abs_dev_v2_raw": max(devs_v2) if devs_v2 else None,
                  "tol_v2_raw": 1e-6,
                  "v2_tier_note": "e254 snapshotted its raw-flavor v2 to "
                                  "fp32 (this cell keeps fp64); the raw "
                                  "flavor compares at the fp32 snapshot "
                                  "tier — an instrument tier, not the "
                                  "reuse claim",
                  "rows": rows_vspan}
            gv["pass"] = bool(devs_p and max(devs_p) <= 1e-12
                              and (not devs_v2 or max(devs_v2) <= 1e-6))
            metrics["gates"].setdefault("G_VSPAN", {})[w] = gv
            journal[w].setdefault("gate_records", {})["G_VSPAN"] = gv
            log(f"G_VSPAN {w}: primary {len(devs_p)} values, max |d| "
                f"{(max(devs_p) if devs_p else float('nan')):.2e} (tol "
                f"1e-12); v2 {len(devs_v2)} values, max |d| "
                f"{(max(devs_v2) if devs_v2 else float('nan')):.2e} "
                f"(tol 1e-6) -> {'PASS' if gv['pass'] else 'FAIL'}")
            journal[w]["vspan_rows"] = rows_vspan

        # ---- release this wash's grads memmap + snapshots
        try:
            gm.flush()
            gm._mmap.close()
        except Exception:                                    # noqa: BLE001
            pass
        for st in list(out["snaps"]):
            out["snaps"][st] = None
            out["snaps2"][st] = None
        out.pop("gm", None)
        (SCRATCH / f"grads_{w}.npy").unlink(missing_ok=True)
        gc.collect()
        journal[w]["reads_done"] = True
        save_journal(rd, journal)
        log(f"{w}: grads memmap + snapshots released")
        write_metrics(f"PARTIAL: {w} reads done")

    # the supports memmap is no longer needed — release (disclosed)
    try:
        sup["mm"].flush()
        sup["mm"]._mmap.close()
    except Exception:                                        # noqa: BLE001
        pass
    sup.pop("mm", None)
    (SCRATCH / "supports.npy").unlink(missing_ok=True)
    gc.collect()
    log("supports memmap released (all qov reads done)")

    # ------------------------------------------- P5 the gates summary
    G_DRAWS = {"per_wash": {w: wash_outputs[w]["gen_ok"] for w in WASHES},
               "pass": bool(all(wash_outputs[w]["gen_ok"] is not False
                                for w in WASHES)) or SMOKE}
    ce_dev = [wash_outputs[w]["ce_cert"] for w in WASHES]
    gn_dev = [wash_outputs[w]["gnorm_cert"] for w in WASHES]
    G_CE = {"max_abs_dce_per_wash": ce_dev,
            "max_abs_dgnorm_per_wash": gn_dev, "tol": 1e-6,
            "pass": bool(all(d is not None and d <= 1e-6 for d in ce_dev)
                         and all(d is not None and d <= 1e-6
                                 for d in gn_dev)) or SMOKE}
    G_REPLAY = {"per_wash": {w: wash_outputs[w]["replay_cert"]
                             for w in WASHES},
                "tol_abs": {"w1": 1e-3, "w2": 1e-2, "w3": 1e-2},
                "note": "w1 CPU-origin ~0 expected; w2 GPU-fp32-origin "
                        "TEXTURE tier ~1.5e-3 (e240/e254's disclosure)"}
    _gr_ok = True
    for w in WASHES:
        for st, rec in (wash_outputs[w]["replay_cert"] or {}).items():
            if isinstance(rec, dict) and "max_abs_dw" in rec:
                if rec["max_abs_dw"] > G_REPLAY["tol_abs"][w]:
                    _gr_ok = False
    G_REPLAY["pass"] = bool(_gr_ok) or SMOKE
    G_MOMENT = {"per_wash": {w: wash_outputs[w]["moment_certs"]
                             for w in WASHES}, "tol": 1e-4,
                "e240_committed_max_rel_v":
                    json.loads(E240_M.read_text(encoding="utf-8"))
                    ["gates"]["G_MOMENT"]["max_rel_v"]}
    G_MOMENT["pass"] = bool(all(
        all(r["rel_v_max"] <= 1e-4 for r in
            (wash_outputs[w]["moment_certs"] or {}).values())
        for w in WASHES)) or SMOKE
    metrics["gates"].update({"G_DRAWS": G_DRAWS, "G_CE": G_CE,
                             "G_REPLAY": G_REPLAY, "G_MOMENT": G_MOMENT})
    if "G_STEPS" in metrics["gates"]:
        metrics["gates"]["G_STEPS"]["pass"] = bool(all(
            r["pass"] for r in metrics["gates"]["G_STEPS"].values()))
    if "G_GRAM" in metrics["gates"]:
        metrics["gates"]["G_GRAM"]["pass"] = bool(all(
            r["pass"] for r in metrics["gates"]["G_GRAM"].values()))
    if "G_VSPAN" in metrics["gates"]:
        metrics["gates"]["G_VSPAN"]["pass"] = bool(all(
            r["pass"] for r in metrics["gates"]["G_VSPAN"].values()))
    log(f"G_DRAWS: {G_DRAWS['per_wash']} | G_CE: dCE {ce_dev} | "
        f"G_MOMENT pass {G_MOMENT['pass']}")
    metrics["all_gates_pass"] = bool(
        all(g.get("pass", False) for g in metrics["gates"].values()
            if isinstance(g, dict))) and not SMOKE
    write_metrics("PARTIAL: replays + reads + gates done")

    if SMOKE:
        metrics["status"] = "SMOKE (nothing adjudicated)"
        write_metrics("SMOKE (nothing adjudicated)")
        log("SMOKE done — nothing adjudicated or gated")
        return 0
    if not metrics["all_gates_pass"]:
        write_metrics("PARTIAL: a gate FAILED — halted before the joins")
        return 1

    # ------------------------------------------------ P6 THE JOINS (the desk)
    load_checks.append(cpu_load_check("joins"))
    thermal_cells = e238m["residual_structure"]["cells"]
    assert len(thermal_cells) == 16

    # e256's row construction, VERBATIM (cell order + canonical battery
    # order over e226's names) — the identical 216 rows
    idx_of = {f: i for i, f in enumerate(names)}
    rows = []
    for cell in thermal_cells:
        w, s_ = cell["wash"], cell["state"]
        b_names = [f for f in names if battery_of[f] == cell["battery"]]
        for f, rz in zip(b_names, cell["resid_z"]):
            rows.append({"wash": w, "state": s_, "fact": f,
                         "battery": battery_of[f], "family6": fam_of[f],
                         "resid_z": rz, "abs_resid_z": abs(rz)})

    # the x-sides
    res256 = {}
    for r in j256["census"]:
        res256[(r["wash"], r["fact"])] = r["residency"]     # A-primary
    for r in rows:
        i = idx_of[r["fact"]]
        qr = journal[r["wash"]]["qov_reads"][str(r["state"])]
        r["qov"] = qr["qov_unit"][i]
        r["qov2"] = qr["qov2_unit"][i]
        r["residency"] = res256[(r["wash"], r["fact"])]
    assert len(rows) == 216

    def sp(xs, ys):
        rho, p = spearmanr(xs, ys)
        return float(rho), float(p)

    join_v = {"statistic": "Spearman(qov_{w,s}, |resid_z|) pooled over "
                           "e238's 16 committed residual cells (the "
                           "identical 216 rows; PRIMARY v = clipped)",
              "flavor": "v PRIMARY (clipped — what the wash's AdamW drank)"}
    join_v["rho"], join_v["p"] = sp([r["qov"] for r in rows],
                                    [r["abs_resid_z"] for r in rows])
    join_v["rho_signed"], join_v["p_signed"] = sp(
        [r["qov"] for r in rows], [r["resid_z"] for r in rows])
    join_v["per_wash"] = {}
    for w in WASHES:
        rw = [r for r in rows if r["wash"] == w]
        join_v["per_wash"][w] = dict(zip(("rho", "p"),
                                         sp([r["qov"] for r in rw],
                                            [r["abs_resid_z"] for r in rw])))
        join_v["per_wash"][w]["n"] = len(rw)
    join_v["per_cell"] = {}
    for w in WASHES:
        for s_ in (50, 80):
            rc = [r for r in rows if r["wash"] == w and r["state"] == s_]
            if rc:
                rr, pp = sp([r["qov"] for r in rc],
                            [r["abs_resid_z"] for r in rc])
                join_v["per_cell"][f"{w}+{s_}"] = {"rho": rr, "p": pp,
                                                    "n": len(rc)}
    join_v["flavor_v2_raw"] = dict(zip(
        ("rho", "p"), sp([r["qov2"] for r in rows],
                         [r["abs_resid_z"] for r in rows])))
    metrics["join_v_overlap"] = join_v

    join_s = {"statistic": "Spearman(residency_w, |resid_z|) on the "
                           "IDENTICAL rows (e256's committed census "
                           "residency, split-A primary) — recomputed here",
              "e256_committed": {"rho": m256["join3_thermal"]["rho"],
                                 "p": m256["join3_thermal"]["p"],
                                 "n_rows": m256["join3_thermal"]["n_rows"]}}
    join_s["rho"], join_s["p"] = sp([r["residency"] for r in rows],
                                    [r["abs_resid_z"] for r in rows])
    join_s["n_rows"] = len(rows)
    join_s["per_wash"] = {}
    for w in WASHES:
        rw = [r for r in rows if r["wash"] == w]
        join_s["per_wash"][w] = dict(zip(("rho", "p"),
                                         sp([r["residency"] for r in rw],
                                            [r["abs_resid_z"] for r in rw])))
        join_s["per_wash"][w]["n"] = len(rw)
    join_s["per_cell"] = {}
    for w in WASHES:
        for s_ in (50, 80):
            rc = [r for r in rows if r["wash"] == w and r["state"] == s_]
            if rc:
                rr, pp = sp([r["residency"] for r in rc],
                            [r["abs_resid_z"] for r in rc])
                join_s["per_cell"][f"{w}+{s_}"] = {"rho": rr, "p": pp,
                                                    "n": len(rc)}
    metrics["join_span_residency"] = join_s

    # G_ROWS: the recomputation == the committed join
    g_rows = {"n_rows": len(rows),
              "d_rho_vs_committed": abs(join_s["rho"]
                                        - m256["join3_thermal"]["rho"]),
              "tol": 1e-12,
              "cells": len(thermal_cells),
              "row_construction": "e256's join-3 construction VERBATIM "
                                  "(cell order + canonical battery order)"}
    g_rows["pass"] = bool(len(rows) == 216 and g_rows["d_rho_vs_committed"]
                          <= 1e-12)
    metrics["gates"]["G_ROWS"] = g_rows
    log(f"G_ROWS: n {len(rows)}, |d rho| {g_rows['d_rho_vs_committed']:.2e} "
        f"-> {'PASS' if g_rows['pass'] else 'FAIL'}")
    metrics["all_gates_pass"] = bool(
        all(g.get("pass", False) for g in metrics["gates"].values()
            if isinstance(g, dict)))
    if not metrics["all_gates_pass"]:
        write_metrics("PARTIAL: G_ROWS FAILED — halted before adjudication")
        return 1

    # THE COMPARISON (the registered bar's arithmetic)
    delta_pool = abs(join_v["rho"]) - abs(join_s["rho"])
    delta_w = {w: abs(join_v["per_wash"][w]["rho"])
               - abs(join_s["per_wash"][w]["rho"]) for w in WASHES}
    min_delta_w = min(delta_w.values())
    discordant = bool((delta_pool >= 0.05 and min_delta_w <= -0.05)
                      or (delta_pool < 0.05 and min_delta_w >= 0.05))
    if delta_pool >= 0.05 and not discordant:
        verdict = "V-OVERLAP-WINS"
    elif delta_pool < 0.05 and not discordant:
        verdict = "SPAN-SPECIFIC"
    else:
        verdict = "MIXED"
    metrics["comparison"] = {
        "delta_pool": delta_pool,
        "delta_per_wash": delta_w,
        "min_delta_w": min_delta_w,
        "discordant": discordant,
        "bar_margin": 0.05,
        "rule_applied": REGISTERED["adjudication_rule_frozen"],
    }
    metrics["adjudication"] = {
        "verdict": verdict,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "the_letter": {
            "V-OVERLAP-WINS": "delta_pool >= 0.05 and not discordant",
            "SPAN-SPECIFIC": "delta_pool < 0.05 and not discordant",
            "MIXED": "discordant"},
        "multiplicity": "one primary join + registered co-report flavors "
                        "(signed, v2-raw, per-wash, per-cell, coarse); NO "
                        "correction claimed (disclosed)",
        "scope": "n=1 organism (GPT-2 124M); the join scope = the y-side's "
                 "committed scope ({w1,w2} x {50,80}; w3 has no residual "
                 "cells); the wash is a repeated measure in pooled tests "
                 "(disclosed); nothing guaranteed",
        "gated_on": "G_ENV/G_SIZE/G_CORPUS/G_BATT/G_ORDER/G_SUPPORT/G_CE/"
                    "G_DRAWS/G_REPLAY/G_MOMENT/G_STEPS/G_BASIS/G_GRAM/"
                    "G_VSPAN/G_ROWS",
        "all_gates_pass": metrics["all_gates_pass"],
    }
    log(f"THE COMPARISON: |rho_v| {abs(join_v['rho']):.4f} vs |rho_span| "
        f"{abs(join_s['rho']):.4f}; delta_pool {delta_pool:+.4f}; "
        f"delta_w {delta_w}; discordant {discordant} -> {verdict}")
    write_metrics(f"PARTIAL: joins done — verdict {verdict}")

    # the coarse battery-level co-report (e256's coarse convention)
    coarse = []
    for fr in e238m["fit_rows"]:
        w = fr["wash"]
        if w not in WASHES:
            continue
        for b, r2 in fr["R2_battery"].items():
            bmed = float(np.median([
                journal[w]["qov_reads"][str(s_)]["qov_unit"][idx_of[f]]
                for s_ in READ_STATES[w]
                for f in names if battery_of[f] == b]))
            coarse.append({"wash": w, "state": fr["state"], "battery": b,
                           "R2": r2, "battery_median_qov": bmed})
    if len(coarse) > 3:
        rr, pp = sp([c["battery_median_qov"] for c in coarse],
                    [c["R2"] for c in coarse])
        metrics["coarse_battery_read"] = {
            "statistic": "Spearman(battery-median qov (pooled over the "
                         "wash's 2 read states, repeated across fit-cell "
                         "states — e256's repetition disclosure), R2_"
                         "battery) over e238's committed fit cells",
            "n": len(coarse), "rho": float(rr), "p": float(pp),
            "e256_committed_coarse_rho":
                m256["join3_thermal"]["coarse_rho"]}
    journal["join_rows"] = rows
    save_journal(rd, journal)

    # --------------------------------------------- P7 the co-reads
    anchors = {}
    prod_band = [f for f in names if fam_of[f] == "product"
                 and f not in (e226.ANCHOR_G, e226.ANCHOR_I)]
    for w in WASHES:
        for s_ in READ_STATES[w]:
            band = np.array([journal[w]["qov_reads"][str(s_)]["qov_unit"]
                             [idx_of[f]] for f in prod_band])
            mu, sd = float(band.mean()), float(band.std(ddof=1))
            key = f"{w}+{s_}"
            anchors[key] = {}
            for tag, f in (("Gmail", e226.ANCHOR_G),
                           ("iPhone", e226.ANCHOR_I)):
                v = float(journal[w]["qov_reads"][str(s_)]["qov_unit"]
                          [idx_of[f]])
                anchors[key][tag] = {"qov": v, "band_mean": mu,
                                     "band_sd": sd,
                                     "z_vs_band": (v - mu) / sd
                                     if sd > 0 else None}
    metrics["anchors"] = {
        "convention": "e239's band: z vs the product family's 5 non-anchor "
                      "probes, per (wash, state)",
        "e256_residency_reference": m256["anchors"],
        "per_state": anchors}

    families = {}
    for w in WASHES:
        for s_ in READ_STATES[w]:
            med = {}
            for fam in ("near-uscap", "product", "cap-cur", "lang",
                        "founder-anchor", "rev-capital"):
                v = [journal[w]["qov_reads"][str(s_)]["qov_unit"]
                     [idx_of[f]] for f in names if fam_of[f] == fam]
                med[fam] = {"n": len(v), "median": float(np.median(v))}
            families[f"{w}+{s_}"] = {
                "medians": med,
                "nearrel_over_product_ratio":
                    med["near-uscap"]["median"] / med["product"]["median"]
                    if med["product"]["median"] > 0 else None}
    metrics["families"] = {
        "e256_t0_residency_reference": {
            w: {"nearrel_over_product_ratio":
                    m256["families"][w]["nearrel_over_product_ratio"]}
            for w in ("w1", "w2", "w3")},
        "per_state": families}

    base_rates = {}
    for w in WASHES:
        for s_ in READ_STATES[w]:
            rec = journal[w]["qov_reads"][str(s_)]
            qv = np.array(rec["qov_unit"])
            nd = np.array(rec["null_draws"])
            base_rates[f"{w}+{s_}"] = {
                "base_rate_mean_v": rec["base_rate_mean_v"],
                "null_band": {"median": float(np.median(nd)),
                              "p05": float(np.percentile(nd, 5)),
                              "p95": float(np.percentile(nd, 95)),
                              "min": float(nd.min()), "max": float(nd.max()),
                              "n_draws": int(nd.shape[0]),
                              "seeds": {"base": NULL_SEED_BASE,
                                        "stride": NULL_SEED_STRIDE,
                                        "chunk_stride":
                                            NULL_CHUNK_STRIDE}},
                "probes": {"median": float(np.median(qv)),
                           "mean": float(qv.mean()),
                           "min": float(qv.min()), "max": float(qv.max()),
                           "median_over_base_rate":
                               float(np.median(qv)
                                     / rec["base_rate_mean_v"]),
                           "n_above_null_p95": int((qv > np.percentile(
                               nd, 95)).sum())},
                "participation_ratio_v": rec["participation_ratio"],
                "note": "E[<u, V u>] for Haar u = mean(v); the band is the "
                        "exact-normalized gaussian-draw distribution; a "
                        "large probe/base-rate ratio is arithmetically "
                        "pre-announced (v is built from squared gradients; "
                        "the supports ARE the organism's gradient "
                        "directions) — the JOIN is the registered read"}
    metrics["base_rates"] = base_rates

    # ------------------------------------------------ P8 the plots
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    bat_col = {"fact": "#8e44ad", "ctrl": "#0d5c3f", "near": "#1a6faf",
               "tmpl": "#b7950b"}
    st_mk = {50: "o", 80: "^"}
    for ax, (xkey, ttl, jr) in zip(
            axes, (("residency",
                    "THE SPAN-RESIDENCY JOIN (e256's carrier)\nrecomputed "
                    "here on the identical rows", join_s),
                   ("qov",
                    "THE SUPPORT-V-OVERLAP JOIN (this cell)\nthe "
                    "denominator's load on the probe's OWN direction",
                    join_v))):
        for b in ("fact", "ctrl", "near", "tmpl"):
            for s_ in (50, 80):
                xs = [r[xkey] for r in rows
                      if r["battery"] == b and r["state"] == s_]
                ys = [r["abs_resid_z"] for r in rows
                      if r["battery"] == b and r["state"] == s_]
                ax.scatter(xs, ys, s=16, alpha=0.6, color=bat_col[b],
                           marker=st_mk[s_],
                           label=b if s_ == 50 else None)
        ax.set_xlabel({"residency": "t=0 span residency (e256 census, "
                                    "split-A)",
                       "qov": r"$\langle s_i,\ V_s\ s_i\rangle/\|s_i\|^2$"
                              " (v PRIMARY, state-level)"}[xkey])
        ax.set_ylabel("|resid_z| (thermal deviation, e238)")
        extra = (f"; per-wash "
                 + ", ".join(f"{w} {jr['per_wash'][w]['rho']:+.3f}"
                             for w in WASHES)) if "per_wash" in jr else ""
        ax.set_title(f"{ttl}\nSpearman rho={jr['rho']:+.4f} "
                     f"p={jr['p']:.4f} (n={jr.get('n_rows', 216)})"
                     f"{extra}  |  o=+50, ^=+80", fontsize=9.5)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle("E257 — THE SUPPORT-V-OVERLAP READ: is the denominator's "
                 "load on the probe's OWN direction the thermal channel's "
                 "carrier?\n"
                 + textwrap.fill(
                     f"delta_pool = |rho_v| - |rho_span| = "
                     f"{delta_pool:+.4f} (bar 0.05) -> {verdict}", 118),
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    png1 = rd / "e257_two_joins.png"
    fig.savefig(png1, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    ax = axes[0][0]
    labels = ["pooled (216)", "w1 (108)", "w2 (108)",
              "w1+50", "w1+80", "w2+50", "w2+80"]
    vals_v = [abs(join_v["rho"])] + \
        [abs(join_v["per_wash"][w]["rho"]) for w in WASHES] + \
        [abs(join_v["per_cell"][k]["rho"])
         for k in ("w1+50", "w1+80", "w2+50", "w2+80")]
    vals_s = [abs(join_s["rho"])] + \
        [abs(join_s["per_wash"][w]["rho"]) for w in WASHES] + \
        [abs(join_s["per_cell"][k]["rho"])
         for k in ("w1+50", "w1+80", "w2+50", "w2+80")]
    xx = np.arange(len(labels))
    ax.bar(xx - 0.19, vals_s, 0.38, color="#5d6d7e", alpha=0.85,
           label="|rho| span-residency (e256's carrier)")
    ax.bar(xx + 0.19, vals_v, 0.38, color="#8e44ad", alpha=0.9,
           label="|rho| support-v-overlap (this cell)")
    ax.set_xticks(xx, labels, fontsize=7.5, rotation=20)
    ax.set_ylabel("|Spearman rho| vs |resid_z|")
    ax.set_title(f"THE HEAD-TO-HEAD — delta_pool {delta_pool:+.4f} "
                 f"(bar 0.05); per-wash deltas "
                 + ", ".join(f"{w} {delta_w[w]:+.3f}" for w in WASHES),
                 fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0][1]
    for key, mk, col in zip(sorted(anchors), "os^v",
                            ("#c0392b", "#2471a3", "#8e44ad", "#0d5c3f")):
        for tag in ("Gmail", "iPhone"):
            ax.scatter([key], [anchors[key][tag]["z_vs_band"]], marker=mk,
                       s=70, color=col if tag == "Gmail" else "k",
                       zorder=3)
            ax.annotate(tag[0], (key, anchors[key][tag]["z_vs_band"]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=9, weight="bold")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(sorted(anchors))), sorted(anchors),
                  rotation=20, fontsize=8)
    ax.set_ylabel("z vs product band (qov)")
    ax.set_title("THE ANCHORS — Gmail/iPhone (marker = state, black = "
                 "iPhone)", fontsize=9.5)
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    fams6 = ("cap-cur", "lang", "founder-anchor", "product", "near-uscap",
             "rev-capital")
    width = 0.8 / (len(families) + 1)
    for fi_, key in enumerate(sorted(families)):
        vals = [families[key]["medians"][fam]["median"] for fam in fams6]
        ax.bar(np.arange(len(fams6)) + fi_ * width - 0.4 + width / 2, vals,
               width=width, alpha=0.75, label=key)
    ax.set_xticks(range(len(fams6)), fams6, rotation=30, fontsize=7.5)
    ax.set_ylabel("median qov")
    rat = ", ".join(f"{k}: {families[k]['nearrel_over_product_ratio']:.2f}"
                    for k in sorted(families))
    r21 = m256["families"]["w1"]["nearrel_over_product_ratio"]
    r22 = m256["families"]["w2"]["nearrel_over_product_ratio"]
    ax.set_title(f"FAMILIES — nearrel vs product (ratios {rat})\n"
                 f"e256 t=0 residency ratios: w1 {r21:.2f}, w2 {r22:.2f}",
                 fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1][1]
    for key, col in zip(sorted(base_rates), ("#8e44ad", "#c0392b",
                                             "#1a6faf", "#0d5c3f")):
        rec = base_rates[key]
        ax.scatter([key], [rec["probes"]["median"]], marker="D", s=70,
                   color=col, zorder=3, label=f"{key} probes' median")
        qv_all = np.array([journal[key.split("+")[0]]["qov_reads"]
                           [str(int(key.split("+")[1]))]["qov_unit"]
                           [idx_of[f]] for f in names])
        ax.scatter(np.full(len(qv_all), key), qv_all, s=8, alpha=0.35,
                   color=col)
        ax.hlines([rec["null_band"]["p05"], rec["null_band"]["p95"]],
                  list(sorted(base_rates)).index(key) - 0.35,
                  list(sorted(base_rates)).index(key) + 0.35,
                  color=col, lw=1.4, ls=":")
        ax.hlines(rec["base_rate_mean_v"],
                  list(sorted(base_rates)).index(key) - 0.35,
                  list(sorted(base_rates)).index(key) + 0.35,
                  color=col, lw=2.2)
    ax.set_yscale("log")
    ax.set_ylabel("quadratic form (log)")
    ax.set_title("THE BASE-RATE DISCLOSURE — probes (dots) vs mean(v) "
                 "(heavy bar) and the random-direction band (dotted p05-p95)",
                 fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.suptitle(f"E257 co-reads — verdict: {verdict}", fontsize=11,
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png2 = rd / "e257_corereads.png"
    fig.savefig(png2, dpi=130)
    plt.close(fig)
    metrics["plot_outputs"] = [str(png1), str(png2)]

    # ------------------------------------------------ P9 closeout
    journal["join_rows"] = rows
    save_journal(rd, journal)
    metrics["honesty_reflex"] = {
        "n": "ONE organism (GPT-2 124M); the join scope = the committed "
             "y-side's scope ({w1,w2} x {50,80}, 216 rows); pooled tests "
             "treat the wash as a repeated measure (disclosed)",
        "provenance": "v is reconstructed by e240/e254's module-imported "
                      "fp64 recursion over the certified replays; certified "
                      "by G_CE/G_DRAWS/G_REPLAY/G_MOMENT/G_STEPS and by "
                      "G_VSPAN against e254's COMMITTED span masses at the "
                      "read states; the y-side and the span census are read "
                      "at runtime from e238's/e256's committed records "
                      "(G_ROWS reproduces e256's committed rho to 1e-12)",
        "circularity": "DISCLOSED and unavoidable in kind: v is built from "
                       "the wash's squared gradients and the supports are "
                       "the organism's gradient directions — the base-rate "
                       "band carries the arithmetic pre-announcement; the "
                       "registered read is the RANK JOIN against an "
                       "independently measured outcome (e238's thermal "
                       "residuals), and the comparison join (span "
                       "residency) shares the same supports",
        "instrument_floor": "the supports' fp16 cache floor (~5e-4 per cos) "
                            "rides every quadratic form; v is fp64",
        "standing_geometry": "the x-side is the t=0 support's overlap with "
                             "the STATE's v (state-level denominator load); "
                             "supports rotate ~18-20 deg through the wash "
                             "(e239's disclosure) — the read is the "
                             "standing direction's load, by registration",
        "multiplicity": "one primary join + registered co-report flavors; "
                        "NO correction claimed",
        "nothing_guaranteed": "MIXED is a real outcome (the registered text "
                              "carries it); n=1 organism",
    }
    metrics["compute"] = {
        "wall_s": round(time.time() - T0, 1),
        "device": "desk (CPU only; torch threads 8 bit-exact convention, "
                  "BLAS 4; no GPU — e248 owns it)",
        "load_checks": len(load_checks),
        "ram_waits": len(ram_waits),
    }
    metrics["trims"] = []
    metrics["deviations"] = deviations
    metrics["status"] = "DONE"
    write_metrics("DONE")
    log(f"done -> {rd}  verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
