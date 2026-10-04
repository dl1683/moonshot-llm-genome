"""E242 — FQ10, THE WALL'S COMMITMENT LAYER (+ its T(t) leg) (eval-only, CPU).

THE QUESTION (FQ10, the ideator's Rank 5, scratch/review_ideator_2026-10-04.md):
every commitment/margin instrument ran at 124M (e228/e230/e232/e238 — the
unwashed organism); the wall (the g-series) has only ever been read through
fact rulers (mean p(Z), the collective dial). Does the 2.74M flat phase hold
margins flat too, or does the commitment grind behind the walled readout?
And T218's registered prediction (a), discharged here as the thermal leg:
does the wall's p-side admit a one-temperature fit — the wall's own T(t)?

THE CELL (the dispatch letter, CPU-only — e237 owns the GPU; e240 shares
the CPU lane, so load-polite: threads 4, load checks, one state = one eval
burst):
  (1) the BEST-INSTRUMENTED wall lineage = g1c's fresh root (e225's J2;
      T181 ROOT-WALL-HOLDS; the freshest committed root with the richest
      committed wash states: runs/checkpoints/g1c_W1_resume.pt carries the
      W1 wash's FULL state dicts at {+1,+2,+4,+10,+50,+100,+200,+300} —
      the g-cell chunked-wash convention). State grid registered:
      {t0, +1, +2, +10, +50, +100, +300}; the two extra committed states
      {+4, +200} ride as TEXTURE co-reports (disclosed; never adjudicated).
  (2) the fact battery's per-probe ARGMAX margins in sigma at every state:
      e228's margin_pass MODULE-IMPORTED (arithmetic untouched), driven
      through e229's thin TinyGPT->.logits adapter shim; the battery =
      the install-60 g-12 ruler (SPLICE_RNG 24301, the family's own —
      e225's roster battery; e229's P0a rebuilt VERBATIM).
  (3) the one-T fit per state on the same battery's p-side: e238's MLE
      instrument PORTED (TempFamily copied VERBATIM from
      lab/e238_temperature_null.py — importing e238 would drag the 124M
      GPT-2 organism's machinery into a 2.74M cell; the port keeps the
      arithmetic byte-identical: same grid, same golden polish, same R2
      definition). The states ARE loadable committed checkpoints, so the
      registered instruction is honored: full answer-position logits are
      re-probed fresh at every state (the two-moment desk bound runs as
      the co-report T218's prediction (b) registered).
  (4) the reads: the ruler (the re-probed g-12 trace, gated bit-close to
      g1c's committed W1 record) vs the margin trajectory vs R2-thermal
      per state — three layers on one page.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any
compute; the script is committed at birth; adjudicate against exactly
this; no bar shopping):
  - FLAT-COMMITMENT — "the ruler flat AND the margins flat through +300
    (both decline < 10%) — the wall holds both layers; a genuine two-layer
    steady state; the shoreline's friction-settle clause weakens to
    readout-only"
  - GRINDING-BEHIND-THE-FLAT — "the ruler flat (its convention) while
    margins fall > 25% by +300 — the wall has zombie decisions DURING the
    flat phase; W036's friction-settle gains its sharpest signature"
  - CROSSOVER — "margins cross the flip zone (0.05 sigma) while the ruler
    still reads — the wall's true failure mode named: it falls at the
    decision layer first"
  - ANY — "the trajectories verbatim, no inflation"
  (T218's thermal leg is a CO-REPORT: the wall's T(t) trajectory + R2,
  quoted against e238's 124M numbers — NO BAR, the comparison is the
  deliverable.)

OPERATIONALIZATIONS (frozen BEFORE compute; they fix the clauses, they do
not move the bars):
  * THE RULER := the install-60 g-12 mean p(Z) (the g-cell's own light
    dial, re-probed on this cell's forwards). "The ruler flat (its
    convention)" := g1c's own committed wall clause: bar_flat = 0.9 x
    root_gm12 holds at every flat-phase checkpoint {10,50,100,300} AND
    the maintain bar (g-12 >= 0.50, G1.MAINTAIN_BAR) holds at every
    transient checkpoint {1,2}. FLAT-COMMITMENT's "(both decline < 10%)"
    is honored as written: it additionally requires the ruler's own +300
    decline (100 x (1 - gm12(+300)/root_gm12)) < 10% — negative decline
    (growth) reads as flat; the committed trace's +300 (0.9406) EXCEEDS
    the root (0.9026), so the two readings coincide on this lineage and
    the conjunction is the honest form.
  * THE MARGINS := per-probe ARGMAX margins in sigma ((top1-top2 logit)/
    std(vocab logits) at the answer position, e228's instrument). The
    aggregate = the battery MEDIAN margin in sigma (e228/e229's own
    primary statistic); co-reports: p25, mean, min, frac below T204's
    ~0.05 sigma flip zone, frac argmax-Z. margin_decline_pct := 100 x
    (1 - median(+300)/median(t0)).
  * CROSSOVER := the battery MEDIAN margin_sigma < 0.05 (inside the flip
    zone) at some registered-grid state s WHILE the ruler still reads
    there (g-12(s) >= 0.50). Co-report: the per-probe frac under 0.05
    crossing 0.5 (the population echo). CROSSOVER adjudicates FIRST (the
    sharpest named mode; a crossing implies the decision layer fell while
    the readout stood).
  * GRINDING-BEHIND-THE-FLAT := ruler flat (its convention, above) AND
    margin_decline_pct > 25. FLAT-COMMITMENT := ruler-flat-convention AND
    ruler decline < 10% AND margin_decline_pct < 10. Composite order
    CROSSOVER / GRINDING / FLAT / ANY (mutually exclusive by
    construction, asserted). If the ruler re-probe FAILS its convention,
    FLAT and GRINDING both stand down honestly (their clauses premise the
    committed flat phase) -> ANY with the trajectories verbatim.
  * THE THERMAL LEG (no bar): per state, ONE T by Bernoulli MLE over the
    60 observed p's under q_i(T) = softmax(L0_i/T)[Z] (L0 = this cell's
    fresh t=0 full-logit re-probe); R2(s) = 1 - sum_i (p_obs,i - q_i)^2
    / sum_i (p_obs,i - p0_i)^2 (T=1 gives R2=0 by construction; e238's
    definition verbatim); LSQ-on-p and full-vocab KL echoes; the
    two-moment desk-only bound (Gaussian denominator from this cell's t0
    (p0, sigma0) triples, N=65); per-probe T_i identifiability spread.
    QUOTED AGAINST e238's committed 124M w1 rows, read at runtime from
    runs/e238/metrics.json and hard-bound (Rule 12).

CHECKS (the dispatch's letter):
  * The states' provenance — bit-exact loads vs the committed g-cell
    records: g1c_root.pt gated by e225's G_ROOT form (param count +
    CPU battery read vs the committed root_gm12, tol 5e-3); every washed
    state's light g-12 re-probe gated against g1c's committed
    adjudication.wall.W1.g_m12 (tol 5e-3); the resume ckpt's final body
    bit-identical to its sds[300] (max|diff| == 0); the battery protocol
    rebuilt from e225's module constants and gated (G_SPLICE 19+41,
    G_BATTERY shapes, G_ANCHOR bank bit-match — e229's P0a VERBATIM).
  * The e208 distinction (T207/W031's convention, restated from e229):
    e208's margin object was the FACT-EDGE over the wash band — that
    instrument died its honest scope death; THIS cell's object is the
    NEXT-TOKEN ARGMAX MARGIN in sigma vs arithmetic noise. A different
    ruler that shares only the beloved word "margin". Not a resurrection.
  * The 2.74M battery dialect: install-60 g-12 ONLY (the family's own
    ruler); NO 124M battery (fact/ctrl/near/tmpl) is computed here — the
    124M numbers enter solely as the thermal leg's quoted comparison
    constants (e238's committed rows, read at runtime, hard-bound).
  * Determinism (T204): same code path, same device -> bit-exact; n=1
    evals suffice; eval batch shape 1 x L disclosed (batch-shape
    re-rounding sits at the ~3e-7 texture floor).
  * Nothing guaranteed.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch;
the GPU lane belongs to e237 — never claimed); torch threads 4 (g1's
import resets to 8 — reset after, g1c's convention); load checks; one
state = one eval burst (60 margin probes + 60 logit forwards + the fits);
minutes total; progressive metrics.json writes after every phase.

Outputs: runs/e242/{metrics.json (PROGRESSIVE), e242_wall_commitment.png,
e242_thermal_leg.png}. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e242_wall_commitment.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e229/e242 convention)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # e228's offline convention (imported)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                     # noqa: E402 — the p25 convention
import torch                                           # noqa: E402
import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402

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
assert not torch.cuda.is_available(), "e242 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------ frozen constants
RUNS = E43.REPO / "runs"
CKPT_DIR = GB.CKPT_DIR
G1C_ROOT_CK = "g1c_root.pt"
G1C_W1_RESUME_CK = "g1c_W1_resume.pt"
G1C_METRICS = RUNS / "g1c_root" / "metrics.json"
E238_METRICS = RUNS / "e238" / "metrics.json"

STATE_GRID: tuple[int, ...] = (1, 2, 10, 50, 100, 300)   # the registered grid (adjudicates)
TEXTURE_STATES: tuple[int, ...] = (4, 200)               # committed; texture co-reports only
FLAT_PHASE_CK = (10, 50, 100, 300)                       # g1c's flat-phase set, intersected w/ grid
TRANSIENT_CK = (1, 2)
G_READ_TOL = 5e-3           # e225/e229's G_ROOT tolerance (CPU read vs committed reads)
T204_FLIP_SIGMA = 0.05      # T204's batch-shape flip threshold — the CROSSOVER zone
MAINTAIN_BAR = float(G1.MAINTAIN_BAR)                    # 0.50, g1/g1b/g1c VERBATIM

# g1c's committed record, HARD-BOUND (every number at its path; Rule 12).
# Read at runtime from runs/g1c_root/metrics.json and asserted against these.
G1C_VERDICT = "ROOT-WALL-HOLDS"
G1C_ROOT_GM12 = 0.9026340246200562                       # root_build.root_cells.gm12 == e225 J2's committed_root_gm12
G1C_W1_GM12 = {                                          # adjudication.wall.W1.g_m12 (the light dial)
    1: 0.821441113948822, 2: 0.9093567132949829, 4: 0.8956068754196167,
    10: 0.9277286529541016, 50: 0.9397175908088684, 100: 0.9368361234664917,
    200: 0.9264556169509888, 300: 0.9405527114868164,
}
G1C_W1_MIN = 0.821441113948822

# e238's committed 124M w1 rows (the thermal leg's comparison constants),
# HARD-BOUND; read at runtime from runs/e238/metrics.json and asserted.
E238_VERDICT = "STRUCTURED"
E238_W1 = {   # step: (T_mle, R2_pooled)
    10: (1.0597956498145238, 0.2992025479329191),
    50: (1.3489839391244727, 0.6258844096157501),
    80: (1.4525923182890421, 0.6896121882948295),
}

REGISTERED_BARS = {
    "FLAT-COMMITMENT": 'FLAT-COMMITMENT — "the ruler flat AND the margins flat '
        'through +300 (both decline < 10%) — the wall holds both layers; a '
        'genuine two-layer steady state; the shoreline\'s friction-settle '
        'clause weakens to readout-only"',
    "GRINDING-BEHIND-THE-FLAT": 'GRINDING-BEHIND-THE-FLAT — "the ruler flat '
        '(its convention) while margins fall > 25% by +300 — the wall has '
        'zombie decisions DURING the flat phase; W036\'s friction-settle '
        'gains its sharpest signature"',
    "CROSSOVER": 'CROSSOVER — "margins cross the flip zone (0.05 sigma) while '
        'the ruler still reads — the wall\'s true failure mode named: it '
        'falls at the decision layer first"',
    "ANY": 'ANY — "the trajectories verbatim, no inflation"',
    "thermal_leg_no_bar": "T218's thermal leg is a CO-REPORT: the wall's T(t) "
                          "trajectory + R2, quoted against e238's 124M "
                          "numbers — no bar, the comparison is the "
                          "deliverable.",
    "registration": "bars frozen VERBATIM from the dispatch brief in the "
                    "module docstring BEFORE any compute (script committed "
                    "at birth); adjudicate against exactly this; no bar "
                    "shopping.",
    "clause_fixes":
        "ruler-flat-convention := g-12 >= 0.9*root_gm12 at {10,50,100,300} "
        "AND g-12 >= 0.50 (G1.MAINTAIN_BAR) at {1,2}, re-probed; margin "
        "aggregate = battery MEDIAN margin_sigma (e228/e229's primary); "
        "margin_decline_pct = 100*(1 - median(+300)/median(t0)); ruler_"
        "decline_pct = 100*(1 - gm12(+300)/root_gm12) (negative = growth "
        "reads as flat); CROSSOVER := exists registered state s with "
        "median_sigma(s) < 0.05 while g-12(s) >= 0.50 — adjudicates FIRST; "
        "GRINDING := ruler-flat-convention AND margin_decline_pct > 25; "
        "FLAT := ruler-flat-convention AND ruler_decline_pct < 10 AND "
        "margin_decline_pct < 10; composite order CROSSOVER / GRINDING / "
        "FLAT / ANY, mutually exclusive by construction; if the ruler "
        "fails its convention FLAT and GRINDING stand down (their clauses "
        "premise the committed flat phase) -> ANY, trajectories verbatim.",
}

deviations: list[str] = [
    "EVAL-ONLY on g1c's committed artifacts (the pristine root + the W1 "
    "resume ckpt's embedded wash states); no training, no wash run here; "
    "CPU-only by dispatch (e237 owns the GPU lane); load-polite vs e240 "
    "(threads 4, load checks, one state = one eval burst).",
    "The thermal instrument is PORTED, not module-imported: TempFamily / "
    "TwoMomentFamily are copied VERBATIM from lab/e238_temperature_null.py "
    "(noted inline at the class) because importing e238 would pull the "
    "124M organism's GPT-2/transformers machinery into a 2.74M CPU cell; "
    "the arithmetic is byte-identical (grid, golden polish, R2 and NLL "
    "definitions unchanged). The margin instrument IS module-imported "
    "(e228.margin_pass via e229's _E228NetShim adapter, arithmetic "
    "untouched).",
    "Importing e228 (via e229) opens runs/e228_run.log in append mode as a "
    "module side effect — nothing is written to it by this cell (own "
    "stdout log).",
    "The state grid: the registered {t0, +1, +2, +10, +50, +100, +300} "
    "adjudicates; the two extra committed states {+4, +200} are read as "
    "TEXTURE co-reports (disclosed here; they never adjudicate — the "
    "frozen clauses name only the registered grid and 'through +300').",
    "The washed states are read through g1's evl_load (settle onto the "
    "ball + disarm — the lineage's registered instrument adaptation), and "
    "certified per-state against g1c's committed W1 light-dial record "
    "(G_STATES, tol 5e-3; observed diffs ~1e-9).",
    "Single battery, single geometry: install-60 g-12 only (the family's "
    "own ruler); no g0/g+12 texture reads, no 124M battery computed here.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).",
    "POST-FIRST-RUN REPORTING FIX (numbers untouched; the e229 precedent): "
    "the first pass's generated FLAT clause printed the margin trajectory's "
    "GROWTH as '-19.2%' — arithmetically right (decline = -19.2%) but "
    "prose-misleading; the clause generator now labels negative declines "
    "as GROWTH explicitly, and the (a)-panel title carries the same note. "
    "Also fixed pre-measurement: the G_STATES_A wall_R check compared "
    "float32-stored 0.7 (0.699999988079071) with exact equality — now "
    "toleranced at 1e-6 (the first pass aborted BEFORE any state "
    "measurement on that gate; no number was ever computed under the "
    "strict form). Deterministic re-run (T204): every number identical; "
    "the verdict and all computed values unchanged; disclosed here rather "
    "than silently patched.",
]


# ------------------------------------------------------------------ the shim
# e229's adapter, reused VERBATIM via module import: TinyGPT(idx) ->
# (logits, loss) becomes the .logits namespace e228's margin_pass expects.
# NO arithmetic lives here.

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
    state_dict order — provenance for the washed states."""
    flat = torch.cat([sd[k].reshape(-1).float()
                      for k in sorted(sd) if not k.startswith("anch__")])
    return hashlib.md5(flat.numpy().tobytes()).hexdigest()


# ------------------------------------------------- the temperature families
# PORTED VERBATIM from lab/e238_temperature_null.py (TempFamily and
# TwoMomentFamily, lines ~452-653) — the dispatch's "e238's MLE instrument,
# ported". Byte-identical arithmetic: same max-shifted softmax, same grid,
# same golden polish, same NLL/R2 definitions. Only the docstrings' frame
# ("54-probe"/"124M") is generic; nothing else touched.

GRID_LO, GRID_HI, GRID_N = -1.0, 1.0, 1201     # log10(T) in [-1, 1] -> T in [0.1, 10]
EPS = 1e-12


class TempFamily:
    """The one-T family on the t=0 dumped logits: q_i(T) = softmax(L0_i/T)[ans_i].

    Everything is float64 numpy; the softmax is max-shifted for stability.
    Precomputes lmax_i and d_i = L0_i - lmax_i once.
    """

    def __init__(self, L0: np.ndarray, ans_ids: np.ndarray):
        self.L0 = L0.astype(np.float64)
        self.ans = ans_ids.astype(np.int64)
        self.lmax = self.L0.max(axis=1)
        self.d = self.L0 - self.lmax[:, None]          # (n, V), <= 0
        self.da = self.d[np.arange(len(self.ans)), self.ans]   # (n,) <= 0
        self.n = len(self.ans)

    def q(self, T: float) -> np.ndarray:
        """The answer probabilities at temperature T (vector over probes)."""
        with np.errstate(over="ignore"):
            num = np.exp(self.da / T)
            den = np.exp(self.d / T).sum(axis=1)
        return num / np.maximum(den, EPS)

    def q_one(self, T: float, i: int) -> float:
        """q at temperature T for probe i only (cheap: one vocab row)."""
        with np.errstate(over="ignore"):
            num = np.exp(self.da[i] / T)
            den = np.exp(self.d[i] / T).sum()
        return float(num / max(den, EPS))

    def grid_Q(self, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
        """One shared grid pass: (log10 T grid, Q (G, n) answer probs)."""
        grid = np.linspace(GRID_LO, GRID_HI, n)
        Q = np.stack([self.q(10.0 ** g) for g in grid])
        return grid, Q

    def _golden(self, f, lo: float, hi: float) -> float:
        gr = (5 ** 0.5 - 1) / 2
        a, b = lo, hi
        for _ in range(80):
            c, d_ = b - gr * (b - a), a + gr * (b - a)
            if f(c) < f(d_):
                b = d_
            else:
                a = c
            if b - a < 1e-8:
                break
        return (a + b) / 2

    def nll_bernoulli(self, T: float, p_obs: np.ndarray) -> float:
        qv = np.clip(self.q(T), EPS, 1.0 - EPS)
        return float(-(p_obs * np.log(qv)
                       + (1.0 - p_obs) * np.log(1.0 - qv)).sum())

    def fit_T_bernoulli(self, p_obs: np.ndarray,
                        grid=None, Q=None) -> tuple[float, float]:
        """MLE over log10 T: shared grid + golden polish. Returns (T*, nll*)."""
        if grid is None or Q is None:
            grid, Q = self.grid_Q()
        Qc = np.clip(Q, EPS, 1.0 - EPS)
        nll_g = -(p_obs * np.log(Qc)
                  + (1.0 - p_obs) * np.log(1.0 - Qc)).sum(axis=1)
        k = int(np.argmin(nll_g))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        xstar = self._golden(
            lambda g: self.nll_bernoulli(10.0 ** g, p_obs), lo, hi)
        return float(10.0 ** xstar), float(
            self.nll_bernoulli(10.0 ** xstar, p_obs))

    def fit_T_lsq(self, p_obs: np.ndarray, grid=None, Q=None) -> float:
        """Least-squares-on-p echo (never adjudicates)."""
        if grid is None or Q is None:
            grid, Q = self.grid_Q()
        k = int(np.argmin(((Q - p_obs) ** 2).sum(axis=1)))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        xstar = self._golden(
            lambda g: float(((self.q(10.0 ** g) - p_obs) ** 2).sum()),
            lo, hi)
        return float(10.0 ** xstar)

    def fit_T_kl_full(self, Lstate: np.ndarray) -> float:
        """Full-vocab KL echo: minimize KL(softmax(Lstate) || softmax(L0/T))
        (one T for the WHOLE distribution — the stronger null; never
        adjudicates)."""
        with np.errstate(over="ignore"):
            qs = np.exp(Lstate - Lstate.max(axis=1, keepdims=True))
            qs = qs / qs.sum(axis=1, keepdims=True)
        qs = np.maximum(qs, 1e-300)
        logqs = np.log(qs)

        def kl(g: float) -> float:
            T = 10.0 ** g
            with np.errstate(over="ignore"):
                z = self.d / T
                z = z - z.max(axis=1, keepdims=True)
                lq = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return float((qs * (logqs - lq)).sum())

        grid = np.linspace(GRID_LO, GRID_HI, 201)
        k = int(np.argmin([kl(g) for g in grid]))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        return float(10.0 ** self._golden(kl, lo, hi))

    def per_probe_T(self, p_obs: np.ndarray,
                    grid=None, Q=None) -> tuple[np.ndarray, np.ndarray]:
        """Per-probe T_i: root of q_i(T) = p_obs,i (grid + bisection).
        Returns (T_i with nan where p_obs is outside the family's reach,
        n_crossings per probe on the grid)."""
        if grid is None or Q is None:
            grid, Q = self.grid_Q()
        n = self.n
        Ti = np.full(n, np.nan)
        ncross = np.zeros(n, dtype=int)
        for i in range(n):
            col = Q[:, i]
            sgn = np.sign(col - p_obs[i])
            cross = np.nonzero(np.diff(sgn) != 0)[0]
            ncross[i] = len(cross)
            if len(cross) == 0:
                continue
            # the crossing nearest T=1 (log10 = 0): grid index k means root
            # in (grid[k], grid[k+1]]
            best = min(cross, key=lambda k: abs(grid[k]))
            a, b = 10.0 ** grid[best], 10.0 ** grid[best + 1]
            fa = col[best] - p_obs[i]
            for _ in range(80):
                m = (a * b) ** 0.5            # geometric midpoint
                fm = self.q_one(m, i) - p_obs[i]
                if np.sign(fm) == np.sign(fa):
                    a, fa = m, fm
                else:
                    b = m
                if b / a - 1 < 1e-9:
                    break
            Ti[i] = (a * b) ** 0.5
        return Ti, ncross


class TwoMomentFamily:
    """The desk-only Gaussian-denominator family from committed triples.

    q_i^2m(T) = 1/[1 + (N-1) exp((mu_i - a_i)/T + sigma0_i^2/(2T^2))] with
    (mu_i - a_i) = ln((1-p0_i)/(p0_i (N-1))) - sigma0_i^2/2, from the t0
    (p0, sigma0) triples. Zero forwards.
    """

    def __init__(self, p0: np.ndarray, sigma0: np.ndarray, N: int):
        self.p0 = np.clip(p0.astype(np.float64), EPS, 1.0 - EPS)
        self.s0 = sigma0.astype(np.float64)
        self.N = int(N)
        self.gap = np.log((1.0 - self.p0) / (self.p0 * (self.N - 1))) \
            - self.s0 ** 2 / 2.0        # mu_i - a_i at T=1

    def q(self, T: float) -> np.ndarray:
        z = self.gap / T + (self.s0 ** 2) / (2.0 * T * T)
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + (self.N - 1) * np.exp(z))

    def nll(self, T: float, p_obs: np.ndarray) -> float:
        qv = np.clip(self.q(T), EPS, 1.0 - EPS)
        return float(-(p_obs * np.log(qv)
                       + (1.0 - p_obs) * np.log(1.0 - qv)).sum())

    def fit_T(self, p_obs: np.ndarray) -> float:
        grid = np.linspace(GRID_LO, GRID_HI, GRID_N)
        k = int(np.argmin([self.nll(10.0 ** g, p_obs) for g in grid]))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, GRID_N - 1)]
        gr = (5 ** 0.5 - 1) / 2
        a, b = lo, hi
        for _ in range(60):
            c, d_ = b - gr * (b - a), a + gr * (b - a)
            if self.nll(10.0 ** c, p_obs) < self.nll(10.0 ** d_, p_obs):
                b = d_
            else:
                a = c
            if b - a < 1e-7:
                break
        return float(10.0 ** ((a + b) / 2))


# ------------------------------------------------- the full-logit re-probe pass

@torch.no_grad()
def logit_pass(net, battery: list[dict]) -> dict:
    """One forward per probe — e228's margin_pass forward path VERBATIM
    (net(ids).logits[0, -1]) plus THIS cell's object on the same logits:
    the FULL answer-position logit vector (fp32) for the one-T family.
    p is recomputed from the same logits and cross-checked against
    margin_pass's p (the internal consistency gate)."""
    net.eval()
    rows, logits = [], []
    for pr in battery:
        lg = net(input_ids=pr["ids"]).logits[0, -1]
        p = F.softmax(lg, -1)
        rows.append({"fact": pr["fact"], "p": float(p[pr["ans_id"]])})
        logits.append(lg.float().numpy().astype(np.float32))
    return {"rows": rows, "logits": np.stack(logits),
            "mean_p": float(np.mean([r["p"] for r in rows]))}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e242")
    metrics: dict = {
        "experiment": "e242_wall_commitment",
        "phase": "FQ10 — the wall's commitment layer: do the 2.74M flat "
                 "phase's margins hold flat too, or does the commitment "
                 "grind behind the walled readout? (+ the wall's own T(t))",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": False,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; the GPU "
                      "lane belongs to e237 — never claimed)",
            "torch_threads": torch.get_num_threads(),
            "bursts": "one state = one eval burst (60 margin probes + 60 "
                      "logit forwards + the fits)",
            "n_evals": "n=1 per state (margins deterministic in-session, T204)",
            "cpu_load_pct_at_launch": cpu_load_probe(),
        },
        "deviations": deviations,
        "builds_on": [
            "FQ10 / review_ideator_2026-10-04 (the Rank-5 question — this "
            "cell's spec)",
            "T218 / e238 (the thermal instrument PORTED; the registered "
            "prediction (a): the wall's batteries admit a T(t) fit; (b): "
            "the two-moment desk bound prices it)",
            "T217 / e234 + T219 / e241 (the commitment-layer vocabulary: "
            "margins, zombies, mis-dials, the frequency collapse)",
            "T208 / e229 + W032 (the margin instrument PORTED TO THE WALL "
            "ROOTS — this cell extends it from the t=0 snapshot to the "
            "TRAJECTORY through the flat phase)",
            "T181 / g1c (the fresh root lineage: ROOT-WALL-HOLDS, the "
            "committed W1 wash states this cell reads)",
            "T209 / e230 + T212 / e232 + W034 (the 124M zombie-decision "
            "narratives this cell asks the wall for)",
            "T204 / x3 (the ~0.05 sigma flip zone + determinism)",
        ],
        "whats_new": [
            "the per-probe argmax-margin TRAJECTORY through the wall's "
            "flat phase (g1c W1, {t0,+1,+2,+10,+50,+100,+300} + {+4,+200} "
            "texture) — margins were only ever read at t=0 on the wall "
            "(e229) or at 124M without a wall (e228/e230/e232)",
            "the one-T fit per state on the WALL's battery p-side — the "
            "wall's own T(t), T218's registered prediction (a) discharged",
            "the three-layer read on one page: the ruler vs the margin "
            "trajectory vs T(t)/R2-thermal, quoted against e238's 124M "
            "numbers",
        ],
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase_note"] = note
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"WROTE partial metrics ({note})")

    log(f"E242 — FQ10, THE WALL'S COMMITMENT LAYER (+ its T(t) leg) -> {rd}")

    # ================= P0a: the battery protocol, e229's P0a VERBATIM ========
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in E225.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E225.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    # g-12 only (the ruler battery; the family's own dialect — see deviations)
    j = E225.RULER_J
    cs = [train_text[p - E225.PRE - j: p] for p, _ in install_occ]
    ruler_ids = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {"g-12": list(ruler_ids.shape)},
        "pass": bool(list(ruler_ids.shape) == [60, E225.PRE - 12]),
        "note": "install-60 g-12 ruler battery ONLY (SPLICE_RNG 24301; the "
                "family's own battery, e225's roster construction re-run "
                "from its module constants; no 124M battery computed here)",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    arng = _random.Random(E225.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - E225.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + E225.BLOCK + 1]
        if any(f in txt for f in E225.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    G_ANCHOR = {"n_windows": 16, "tries": tries,
                "bank_starts_match_e185_stored":
                    bool(n_starts == E225.E170_BANK_STARTS)}
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR}
    log("P0a: protocol gates PASS (e229's P0a ported; splice 19+41; g-12 "
        "shapes; e170 bank bit-match)")
    write_partial("P0a protocol gates PASSED")

    # ================= P0b: G_PARENTS — the committed records hard-bound ======
    if not G1C_METRICS.exists():
        raise RuntimeError(f"missing parent record: {G1C_METRICS}")
    if not E238_METRICS.exists():
        raise RuntimeError(f"missing parent record: {E238_METRICS}")
    g1c = json.loads(G1C_METRICS.read_text(encoding="utf-8"))
    e238m = json.loads(E238_METRICS.read_text(encoding="utf-8"))

    g1c_w1 = {int(k): v for k, v in
              g1c["adjudication"]["wall"]["W1"]["g_m12"].items()}
    g1c_root = g1c["root_build"]["root_cells"]["gm12"]
    ok_g1c = (g1c["adjudication"]["verdict"] == G1C_VERDICT
              and abs(g1c_root - G1C_ROOT_GM12) < 1e-12
              and all(abs(g1c_w1[s] - G1C_W1_GM12[s]) < 1e-12 for s in G1C_W1_GM12)
              and abs(g1c["adjudication"]["wall"]["W1"]["min_gm12"]
                      - G1C_W1_MIN) < 1e-12)
    # e225's J2 row cross-check (the roster's own committed root read)
    j2 = next(r for r in E225.ROSTER if r["id"] == "J2")
    ok_g1c &= (j2["ckpt"] == G1C_ROOT_CK
               and j2["n_params"] == 2_739_072
               and abs(j2["committed_root_gm12"] - G1C_ROOT_GM12) < 1e-12)

    e238_w1 = {}
    for r in e238m["fit_rows"]:
        if r["wash"] == "w1" and r["state"] in E238_W1:
            e238_w1[r["state"]] = (r["T_mle"], r["R2_pooled"])
    ok_e238 = (e238m["adjudication"]["bars"]["verdict"] == E238_VERDICT
               and set(e238_w1) == set(E238_W1)
               and all(abs(e238_w1[s][0] - E238_W1[s][0]) < 1e-12
                       and abs(e238_w1[s][1] - E238_W1[s][1]) < 1e-12
                       for s in E238_W1))

    G_PARENTS = {
        "g1c_metrics": {"path": str(G1C_METRICS), "md5": md5of(G1C_METRICS),
                        "sha256_16": sha256_of(G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"],
                        "root_gm12": g1c_root,
                        "W1_g_m12": {str(k): v for k, v in sorted(g1c_w1.items())},
                        "W1_min_gm12": g1c["adjudication"]["wall"]["W1"]["min_gm12"]},
        "e238_metrics": {"path": str(E238_METRICS), "md5": md5of(E238_METRICS),
                         "sha256_16": sha256_of(E238_METRICS),
                         "verdict": e238m["adjudication"]["bars"]["verdict"],
                         "w1_T_R2_quoted": {str(s): {"T_mle": e238_w1[s][0],
                                                     "R2_pooled": e238_w1[s][1]}
                                            for s in sorted(e238_w1)}},
        "pass": bool(ok_g1c and ok_e238),
        "note": "g1c's committed record (the y-side ruler + the state "
                "certification targets) and e238's committed 124M w1 rows "
                "(the thermal leg's comparison constants) read at runtime "
                "and asserted against the literals frozen in this script "
                "pre-compute (Rule 12)",
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — g1c {G1C_VERDICT} (root {g1c_root:.6f}, W1 "
        f"+300 {g1c_w1[300]:.6f}); e238 {E238_VERDICT} (w1 T "
        + "/".join(f"{e238_w1[s][0]:.3f}" for s in sorted(e238_w1)) + ")")
    write_partial("P0b parent gates PASSED")

    # ================= P1: the states — load, gate, measure ==================
    # t0 = the pristine root (e225's load_body convention); washed states =
    # the W1 resume ckpt's embedded sds, read through g1's evl_load
    # (settle+disarm, the lineage's registered instrument adaptation).
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
    G_STATES_A = {
        "checkpoint": f"runs/checkpoints/{G1C_W1_RESUME_CK}",
        "sha256_16": sha256_of(CKPT_DIR / G1C_W1_RESUME_CK),
        "step_field": int(wck["step"]),
        "wall_R": float(wck["wall_R"]),
        "sds_steps": sds_keys,
        "sds300_bit_identical_to_final_model": bool(max_diff_300 == 0.0),
        "expected_sds_steps": list(all_steps),
        "wash_seed_field": None,   # filled from g1c's config below
        "pass": bool(int(wck["step"]) == 300
                     and abs(float(wck["wall_R"]) - 0.7) < 1e-6   # fp32 round-trip of 0.7 = 0.699999988079071
                     and tuple(sds_keys) == all_steps and max_diff_300 == 0.0),
    }
    assert G_STATES_A["pass"], f"W1 resume inventory gate FAILED: {G_STATES_A}"

    # the t0 root — e225's G_ROOT form (param count + battery read)
    root_net, root_sd, root_meta = E225.load_body(CKPT_DIR / G1C_ROOT_CK, Cfg())
    n_par = sum(p.numel() for p in root_net.parameters())
    theta0 = E225.flat_params(root_net)
    root_read = E225.battery_cell(root_net, ruler_ids, zid)["mean_pz"]
    rdev = abs(root_read - G1C_ROOT_GM12)
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{G1C_ROOT_CK}",
        "n_params": n_par, "expected_params": 2_739_072,
        "battery_read_measured": root_read,
        "battery_read_committed": G1C_ROOT_GM12,
        "abs_diff": rdev, "tol": G_READ_TOL,
        "flat_md5": hashlib.md5(theta0.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == 2_739_072 and rdev < G_READ_TOL),
    }
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    log(f"P1: G_ROOT[{G1C_ROOT_CK}] PASS — {n_par} params; battery "
        f"{root_read:.10f} vs committed {G1C_ROOT_GM12:.10f} "
        f"(|d| {rdev:.1e})")
    metrics["gates"]["G_ROOT"] = G_ROOT
    metrics["gates"]["G_STATES_A"] = G_STATES_A

    def probe_battery(ids: torch.Tensor) -> list[dict]:
        return [{"ids": ids[i: i + 1],
                 "fact": f"install{i:02d}@g-12",
                 "relation": "install60_g-12",
                 "ans_id": zid}
                for i in range(ids.shape[0])]

    pbatt = probe_battery(ruler_ids)

    def measure_state(tag: str, net) -> dict:
        """One state = one eval burst: e228's margin_pass (module-imported,
        via e229's shim) + the full-logit re-probe pass + the internal
        consistency cross-check."""
        loadpct = cpu_load_probe()
        shim = E229._E228NetShim(net)
        mrec = E228.margin_pass(shim, pbatt)
        lrec = logit_pass(shim, pbatt)
        dp = max(abs(r["p"] - m["p"])
                 for r, m in zip(lrec["rows"], mrec["probes"]))
        ms = [r["margin_sigma"] for r in mrec["probes"]]
        # the light dial on the SAME net (the g-cell ruler, for the gate)
        gm12 = G1.battery_cell(net, ruler_ids, zid)["mean_pz"]
        cell = {
            "tag": tag, "cpu_load_pct": loadpct,
            "gm12_ruler": gm12,
            "margin_aggregate": {
                "median": mrec["median_margin_sigma"],
                "mean": mrec["mean_margin_sigma"],
                "p25": float(np.percentile(ms, 25)),
                "min": mrec["min_margin_sigma"],
                "frac_below_T204_flip_zone":
                    float(sum(1 for m in ms if m < T204_FLIP_SIGMA) / len(ms)),
                "frac_argmax_z": float(sum(
                    1 for r in mrec["probes"] if r["top1_id"] == zid) / len(ms)),
            },
            "mean_p": mrec["mean_p"],
            "logit_pass_max_dp_vs_margin_pass": dp,
            "probes": [{"fact": r["fact"], "p": r["p"],
                        "margin_sigma": r["margin_sigma"],
                        "margin_raw": r["margin_raw"], "sigma": r["sigma"],
                        "top1_id": r["top1_id"], "top2_id": r["top2_id"],
                        "argmax_is_z": bool(r["top1_id"] == zid)}
                       for r in mrec["probes"]],
            "logits": lrec["logits"],      # (60, V) fp32 — the thermal family's base/target
        }
        return cell

    cells: dict = {}

    # t0
    cells["t0"] = measure_state("t0", root_net)
    cells["t0"]["G"] = {"kind": "pristine root (e225 load_body)",
                        "flat_md5": G_ROOT["flat_md5"]}
    log(f"  t0: ruler {cells['t0']['gm12_ruler']:.4f} | median margin "
        f"{cells['t0']['margin_aggregate']['median']:.4f}s p25 "
        f"{cells['t0']['margin_aggregate']['p25']:.4f}s argmax-Z "
        f"{cells['t0']['margin_aggregate']['frac_argmax_z']:.2f} | "
        f"<0.05s {cells['t0']['margin_aggregate']['frac_below_T204_flip_zone']:.2f}")
    write_partial("P1 t0 measured")
    del root_net

    # the washed states
    g_state_rows = {}
    for s in all_steps:
        net = G1.evl_load(wck["sds"][s])
        key = f"w1+{s}"
        cells[key] = measure_state(key, net)
        cells[key]["G"] = {"kind": "W1 wash state (g1 evl_load settle+disarm)",
                           "body_flat_md5": body_flat_md5(wck["sds"][s])}
        committed = G1C_W1_GM12[s]
        d = abs(cells[key]["gm12_ruler"] - committed)
        g_state_rows[s] = {"measured": cells[key]["gm12_ruler"],
                           "committed": committed, "abs_diff": d}
        role = "ADJUDICATES" if s in STATE_GRID else "texture"
        log(f"  +{s:>3} ({role}): ruler {cells[key]['gm12_ruler']:.6f} vs "
            f"committed {committed:.6f} (|d| {d:.1e}) | median margin "
            f"{cells[key]['margin_aggregate']['median']:.4f}s | argmax-Z "
            f"{cells[key]['margin_aggregate']['frac_argmax_z']:.2f}")
        del net
        write_partial(f"P1 w1+{s} measured")

    G_STATES = {
        **{k: v for k, v in G_STATES_A.items()},
        "per_state_ruler_reads": {str(k): v for k, v in sorted(g_state_rows.items())},
        "tol": G_READ_TOL,
        "max_abs_diff": max(v["abs_diff"] for v in g_state_rows.values()),
        "pass": bool(G_STATES_A["pass"] and all(
            v["abs_diff"] < G_READ_TOL for v in g_state_rows.values())),
        "note": "every W1 state's light g-12 re-probe (this cell's forwards) "
                "vs g1c's committed adjudication.wall.W1.g_m12 — the "
                "bit-exactness certification of the loaded states (observed "
                "diffs ~1e-9; the tol is e225's G_ROOT convention)",
    }
    assert G_STATES["pass"], f"state certification FAILED: {G_STATES}"
    metrics["gates"]["G_STATES"] = G_STATES
    # strip the raw logits + full probe rows from the cells record (journal
    # economy; the aggregates + per-probe margins stay committed)
    cells_light = {}
    for k, c in cells.items():
        cc = {kk: vv for kk, vv in c.items()
              if kk not in ("logits", "probes")}
        cc["probes"] = [{"fact": p["fact"], "p": p["p"],
                         "margin_sigma": p["margin_sigma"],
                         "argmax_is_z": p["argmax_is_z"],
                         "top1_id": p["top1_id"], "top2_id": p["top2_id"]}
                        for p in c["probes"]]
        cells_light[k] = cc
    metrics["cells"] = cells_light
    log(f"P1 COMPLETE: all {len(all_steps)} W1 states + t0 gated (max |d| "
        f"{G_STATES['max_abs_diff']:.1e}) and margin-passed")
    write_partial("P1 COMPLETE (all states gated + margin-passed)")

    # ================= P2: the trajectory + the frozen adjudication ==========
    traj = []
    for key in ["t0"] + [f"w1+{s}" for s in all_steps]:
        c = cells[key]
        step = 0 if key == "t0" else int(key.split("+")[1])
        traj.append({
            "state": key, "step": step,
            "role": "t0" if step == 0 else
                    ("ADJUDICATES" if step in STATE_GRID else "texture"),
            "gm12_ruler": c["gm12_ruler"],
            "margin_median_sigma": c["margin_aggregate"]["median"],
            "margin_p25_sigma": c["margin_aggregate"]["p25"],
            "margin_mean_sigma": c["margin_aggregate"]["mean"],
            "margin_min_sigma": c["margin_aggregate"]["min"],
            "frac_below_flip_zone": c["margin_aggregate"]["frac_below_T204_flip_zone"],
            "frac_argmax_z": c["margin_aggregate"]["frac_argmax_z"],
            "mean_p": c["mean_p"],
        })
    metrics["trajectory"] = traj

    root_gm12 = cells["t0"]["gm12_ruler"]
    med0 = cells["t0"]["margin_aggregate"]["median"]
    med300 = cells["w1+300"]["margin_aggregate"]["median"]
    gm12_300 = cells["w1+300"]["gm12_ruler"]
    margin_decline_pct = 100.0 * (1.0 - med300 / med0) if med0 > 0 else float("nan")
    ruler_decline_pct = 100.0 * (1.0 - gm12_300 / root_gm12)
    ruler_flat_convention = (
        all(cells[f"w1+{s}"]["gm12_ruler"] >= 0.9 * root_gm12
            for s in FLAT_PHASE_CK)
        and all(cells[f"w1+{s}"]["gm12_ruler"] >= MAINTAIN_BAR
                for s in TRANSIENT_CK))
    crossover_rows = [
        {"state": f"w1+{s}", "median_sigma": cells[f"w1+{s}"]["margin_aggregate"]["median"],
         "gm12": cells[f"w1+{s}"]["gm12_ruler"]}
        for s in STATE_GRID
        if cells[f"w1+{s}"]["margin_aggregate"]["median"] < T204_FLIP_SIGMA
        and cells[f"w1+{s}"]["gm12_ruler"] >= MAINTAIN_BAR]

    crossover_fires = len(crossover_rows) > 0
    grinding_fires = bool(ruler_flat_convention and margin_decline_pct > 25.0)
    flat_fires = bool(ruler_flat_convention and ruler_decline_pct < 10.0
                      and margin_decline_pct < 10.0)
    any_fires = not (crossover_fires or grinding_fires or flat_fires)
    assert (int(crossover_fires) + int(grinding_fires) + int(flat_fires)
            + int(any_fires)) == 1, "bars must be exclusive"

    def _decl_txt(pct: float) -> str:
        """Honest decline formatting: a negative decline IS growth (the
        frozen clauses read growth as flat; the prose must say so)."""
        return (f"{pct:+.1f}% "
                + ("(GROWTH — reads as flat under the frozen clause)"
                   if pct < 0 else "(decline)"))

    if crossover_fires:
        verdict = "CROSSOVER"
        r0 = crossover_rows[0]
        clause = (f"margins cross the flip zone: the battery MEDIAN margin "
                  f"falls under {T204_FLIP_SIGMA} sigma at {r0['state']} "
                  f"({r0['median_sigma']:.4f}s) while the ruler still reads "
                  f"(g-12 {r0['gm12']:.3f} >= {MAINTAIN_BAR}) — the wall's "
                  f"true failure mode named: it falls at the decision layer "
                  f"first (margin decline through +300 "
                  f"{_decl_txt(margin_decline_pct)}; ruler "
                  f"{_decl_txt(ruler_decline_pct)})")
    elif grinding_fires:
        verdict = "GRINDING-BEHIND-THE-FLAT"
        clause = (f"the ruler flat (its convention: every flat-phase ckpt "
                  f">= 0.9 x root {0.9 * root_gm12:.4f} and every transient "
                  f">= {MAINTAIN_BAR}) while margins fall "
                  f"{abs(margin_decline_pct):.1f}% by +300 "
                  f"(median {med0:.4f}s -> {med300:.4f}s) — the wall has "
                  f"zombie decisions DURING the flat phase; W036's "
                  f"friction-settle gains its sharpest signature")
    elif flat_fires:
        clause = (f"the ruler flat AND the margins flat through +300: the "
                  f"ruler {root_gm12:.4f} -> {gm12_300:.4f} "
                  f"{_decl_txt(ruler_decline_pct)} and the margin median "
                  f"{med0:.4f}s -> {med300:.4f}s "
                  f"{_decl_txt(margin_decline_pct)}, both under the 10% "
                  f"decline bar — the wall holds both layers; a genuine "
                  f"two-layer steady state; the shoreline's friction-settle "
                  f"clause weakens to readout-only")
        verdict = "FLAT-COMMITMENT"
    else:
        why = []
        if not ruler_flat_convention:
            why.append(f"the ruler FAILS its own convention on the re-probe "
                       f"(the committed flat phase does not reproduce)")
        else:
            why.append(f"the ruler holds its convention but the margin "
                       f"decline {margin_decline_pct:+.1f}% lands between "
                       f"the bars (10% < decline <= 25%)")
        clause = ("; ".join(why) + " — the trajectories verbatim, no "
                  "inflation")
        verdict = "ANY"

    gates_summary = {}
    for g, v in metrics["gates"].items():
        gates_summary[g] = bool(v.get("pass")) if isinstance(v, dict) else bool(v)
    metrics["adjudication"] = {
        "bars": {"FLAT-COMMITMENT": {"fires": flat_fires},
                 "GRINDING-BEHIND-THE-FLAT": {"fires": grinding_fires},
                 "CROSSOVER": {"fires": crossover_fires},
                 "ANY": {"fires": any_fires}},
        "clause_fixes_applied": REGISTERED_BARS["clause_fixes"],
        "verdict": verdict, "clause": clause,
        "reads": {
            "root_gm12": root_gm12,
            "gm12_300": gm12_300,
            "ruler_decline_pct": ruler_decline_pct,
            "ruler_flat_convention": ruler_flat_convention,
            "bar_flat_line": 0.9 * root_gm12,
            "margin_median_t0": med0,
            "margin_median_300": med300,
            "margin_decline_pct": margin_decline_pct,
            "crossover_cells": crossover_rows,
            "flip_zone_sigma": T204_FLIP_SIGMA,
        },
        "composite_order": "CROSSOVER / GRINDING-BEHIND-THE-FLAT / "
                           "FLAT-COMMITMENT / ANY (frozen before compute; "
                           "mutually exclusive by construction)",
        "gates_summary": gates_summary,
    }
    log("=" * 78)
    log(f"E242 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)
    write_partial("P2 the frozen bars adjudicated")

    # ================= P3: the thermal leg (co-report, NO bar) ===============
    # the family's base = THIS cell's fresh t0 full-logit dump; every state's
    # observed p's = this cell's logit_pass p's (cross-checked against
    # margin_pass above). e238's instrument, ported verbatim.
    L0 = cells["t0"]["logits"]
    p0 = np.array([r["p"] for r in cells["t0"]["probes"]])
    sig0 = np.array([r["sigma"] for r in cells["t0"]["probes"]])
    V = int(L0.shape[1])
    fam = TempFamily(L0, np.full(len(p0), zid, dtype=np.int64))
    tm2 = TwoMomentFamily(p0, sig0, V)
    grid, Q = fam.grid_Q()

    thermal_rows = []
    for key in ["t0"] + [f"w1+{s}" for s in all_steps]:
        c = cells[key]
        step = 0 if key == "t0" else int(key.split("+")[1])
        p_obs = np.array([r["p"] for r in c["probes"]])
        Lst = c["logits"]
        T_mle, nll = fam.fit_T_bernoulli(p_obs, grid, Q)
        q = fam.q(T_mle)
        ss_res = float(((p_obs - q) ** 2).sum())
        ss_tot = float(((p_obs - p0) ** 2).sum())
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        T_lsq = fam.fit_T_lsq(p_obs, grid, Q)
        T_kl = fam.fit_T_kl_full(Lst)
        T_2m = tm2.fit_T(p_obs)
        q_2m = np.clip(tm2.q(T_2m), EPS, 1.0 - EPS)
        ss_res_2m = float(((p_obs - q_2m) ** 2).sum())
        R2_2m = 1.0 - ss_res_2m / ss_tot if ss_tot > 0 else float("nan")
        Ti, ncross = fam.per_probe_T(p_obs, grid, Q)
        ti_ok = Ti[~np.isnan(Ti)]
        thermal_rows.append({
            "state": key, "step": step,
            "role": "t0" if step == 0 else
                    ("ADJUDICATES" if step in STATE_GRID else "texture"),
            "T_mle": T_mle, "nll": nll, "R2_pooled": R2,
            "T_lsq": T_lsq, "T_kl_full": T_kl,
            "T_two_moment": T_2m, "R2_two_moment": R2_2m,
            "ss_tot_decline": ss_tot,
            "degenerate_denominator": bool(ss_tot < 1e-3),
            "mean_p_obs": float(p_obs.mean()), "mean_q": float(q.mean()),
            "per_probe_T_i": {
                "n_identified": int(len(ti_ok)),
                "median": float(np.median(ti_ok)) if len(ti_ok) else None,
                "iqr_over_median": (float((np.percentile(ti_ok, 75)
                                           - np.percentile(ti_ok, 25))
                                          / np.median(ti_ok)))
                if len(ti_ok) and np.median(ti_ok) > 0 else None,
            },
        })
        log(f"  thermal {key:>7}: T {T_mle:.4f} R2 {R2:+.4f} (LSQ {T_lsq:.3f} "
            f"KL {T_kl:.3f} 2m {T_2m:.3f}; T_i n={len(ti_ok)}; decline-SS "
            f"{ss_tot:.4f}{' — DEGENERATE' if ss_tot < 1e-3 else ''})")
    metrics["thermal_leg"] = {
        "no_bar": REGISTERED_BARS["thermal_leg_no_bar"],
        "family": "q_i(T) = softmax(L0_i/T)[Z]; L0 = this cell's fresh t0 "
                  "full-logit re-probe (e238's instrument PORTED verbatim); "
                  "ONE T per state; Bernoulli MLE over the 60 observed p's",
        "rows": thermal_rows,
        "e238_124M_quote": {
            "source": str(E238_METRICS),
            "md5": md5of(E238_METRICS),
            "verdict": E238_VERDICT,
            "w1": {str(s): {"T_mle": E238_W1[s][0], "R2_pooled": E238_W1[s][1]}
                   for s in sorted(E238_W1)},
            "note": "the 124M unwalled wash's own T(t) — the wall's T(t) is "
                    "quoted AGAINST this, never adjudicated",
        },
        "comparison_read": None,   # filled below
    }
    # the wall-vs-124M comparison sentence (computed, quoted verbatim)
    w_T = {r["step"]: r["T_mle"] for r in thermal_rows if r["step"] in STATE_GRID}
    w_R2 = {r["step"]: r["R2_pooled"] for r in thermal_rows if r["step"] in STATE_GRID}
    metrics["thermal_leg"]["comparison_read"] = (
        f"the wall's T(t): 1.000 at t0 -> "
        + " -> ".join(f"{w_T[s]:.4f} (+{s})" for s in STATE_GRID)
        + f"; R2 {w_R2[300]:+.3f} at +300 (vs e238's 124M unwalled w1: T "
        f"1.06 -> 1.35 -> 1.45 over +10/+50/+80 with R2 0.30 -> 0.63 -> "
        f"0.69) — the wall's p-side "
        + ("barely moves" if max(abs(w_T[s] - 1.0) for s in STATE_GRID) < 0.1
           else "does move; quoted verbatim"))
    log(f"P3 thermal leg: " + metrics["thermal_leg"]["comparison_read"])
    write_partial("P3 the thermal leg computed")

    # ================= P4: the figures ========================================
    steps_all = [r["step"] for r in traj]
    xs = np.arange(len(traj))

    # ---- main figure: the ruler-vs-margin-vs-T trajectories
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 7.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})

    # (a) the ruler vs the margin aggregate (twin axes)
    ax = axes[0]
    ys_ruler = [r["gm12_ruler"] for r in traj]
    ys_med = [r["margin_median_sigma"] for r in traj]
    adj_idx = [i for i, r in enumerate(traj) if r["role"] != "texture"]
    tex_idx = [i for i, r in enumerate(traj) if r["role"] == "texture"]
    ax.plot(xs, ys_ruler, "-", color="black", lw=1.6, alpha=0.85, zorder=3)
    ax.plot([xs[i] for i in adj_idx], [ys_ruler[i] for i in adj_idx], "o",
            ms=9, color="black", zorder=4, label="ruler g-12 mean p(Z) (left)")
    ax.plot([xs[i] for i in tex_idx], [ys_ruler[i] for i in tex_idx], "o",
            ms=7, mfc="white", mec="black", zorder=4)
    ax.axhline(0.9 * root_gm12, ls="--", lw=1.1, color="gray")
    ax.axhline(MAINTAIN_BAR, ls=":", lw=1.1, color="gray")
    ax.annotate(f"bar_flat 0.9xroot = {0.9 * root_gm12:.4f}",
                (0.02, 0.9 * root_gm12), xycoords=("axes fraction", "data"),
                fontsize=7, color="gray", va="bottom")
    ax.annotate(f"maintain {MAINTAIN_BAR:.2f}", (0.02, MAINTAIN_BAR),
                xycoords=("axes fraction", "data"), fontsize=7,
                color="gray", va="bottom")
    ax.set_xlabel("wash step (W1, the wall R=0.7; open = texture states)")
    ax.set_ylabel("THE RULER — install-60 g-12 mean p(Z)")
    ax.set_ylim(0.0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(xs, ys_med, "-", color="tab:red", lw=1.8, zorder=3)
    ax2.plot([xs[i] for i in adj_idx], [ys_med[i] for i in adj_idx], "s",
             ms=9, color="tab:red", zorder=4,
             label="margin MEDIAN sigma (right)")
    ax2.plot([xs[i] for i in tex_idx], [ys_med[i] for i in tex_idx], "s",
             ms=7, mfc="white", mec="tab:red", zorder=4)
    ax2.axhline(T204_FLIP_SIGMA, ls="--", lw=1.2, color="tab:red", alpha=0.6)
    ax2.annotate("T204 flip zone 0.05s", (0.35, T204_FLIP_SIGMA),
                 xycoords=("axes fraction", "data"), fontsize=7,
                 color="tab:red", va="bottom")
    ax2.set_ylabel("THE COMMITMENT — battery median argmax margin (sigma)",
                   color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")
    ax.set_xticks(xs)
    ax.set_xticklabels(["t0"] + [f"+{s}" for s in steps_all[1:]], fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(f"(a) THE TWO LAYERS — ruler {ruler_decline_pct:+.1f}% vs "
                 f"margin median {margin_decline_pct:+.1f}% through +300 "
                 f"(negative = GROWTH)", fontsize=9.5)

    # (b) normalized declines
    ax = axes[1]
    ys_ruler_n = [y / root_gm12 for y in ys_ruler]
    ys_med_n = [y / med0 for y in ys_med]
    ys_p25 = [r["margin_p25_sigma"] / cells["t0"]["margin_aggregate"]["p25"]
              for r in traj]
    ys_frac = [r["frac_argmax_z"] for r in traj]
    ax.axhspan(0.90, 1.10, color="gold", alpha=0.15)
    ax.axhline(0.90, color="k", ls="--", lw=1.1, label="the 10% decline line")
    ax.axhline(0.75, color="crimson", ls="--", lw=1.1,
               label="the 25% grinding line")
    ax.plot(xs, ys_ruler_n, "o-", color="black", ms=7, lw=1.6,
            label="ruler / ruler(t0)")
    ax.plot(xs, ys_med_n, "s-", color="tab:red", ms=7, lw=1.6,
            label="margin median / median(t0)")
    ax.plot(xs, ys_p25, "^:", color="darkorange", ms=5, lw=1.1,
            label="margin p25 / p25(t0)")
    ax.plot(xs, ys_frac, "v-.", color="tab:blue", ms=5, lw=1.1, alpha=0.8,
            label="frac argmax-Z (raw)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["t0"] + [f"+{s}" for s in steps_all[1:]], fontsize=8)
    ax.set_ylabel("normalized to t0 (frac argmax-Z raw, right-ish)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.2, loc="best")
    ax.set_title("(b) THE COMMITMENT TRAJECTORY — both decline lines drawn "
                 "from the frozen bars", fontsize=9.5)

    # (c) the thermal leg: the wall's T(t) vs e238's 124M
    ax = axes[2]
    th_t0 = thermal_rows[0]
    th_steps = [r["step"] for r in thermal_rows]
    th_T = [r["T_mle"] for r in thermal_rows]
    th_adj = [i for i, r in enumerate(thermal_rows) if r["role"] != "texture"]
    th_tex = [i for i, r in enumerate(thermal_rows) if r["role"] == "texture"]
    ax.plot(np.arange(len(thermal_rows)), th_T, "D-", color="tab:purple",
            lw=1.8, label="THE WALL's T(t) (this cell)")
    ax.plot(th_adj, [th_T[i] for i in th_adj], "D", ms=8, color="tab:purple")
    ax.plot(th_tex, [th_T[i] for i in th_tex], "D", ms=6, mfc="white",
            mec="tab:purple")
    ax.plot([0.1, 1.2, 4.2, 7.0], [1.0] + [E238_W1[s][0] for s in (10, 50, 80)],
            "s--", color="tab:gray", lw=1.4, ms=7,
            label="e238's 124M UNWALLED w1 T(t) (quoted)")
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_xticks(np.arange(len(thermal_rows)))
    ax.set_xticklabels(["t0"] + [f"+{s}" for s in th_steps[1:]], fontsize=8)
    ax.set_xlabel("wash step (124M curve: +10/+50/+80 mapped to the same "
                  "axis slots for shape comparison)")
    ax.set_ylabel("fitted T (one-T MLE)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")
    r2_300 = [r for r in thermal_rows if r["step"] == 300][0]["R2_pooled"]
    ax.set_title(f"(c) THE THERMAL LEG (no bar) — T218's prediction (a): "
                 f"the wall's p-side DOES admit a T(t); R2(+300) "
                 f"{r2_300:+.3f} vs e238 +80 {E238_W1[80][1]:+.3f}",
                 fontsize=9.0)

    fig.suptitle("E242 — FQ10, THE WALL'S COMMITMENT LAYER: the ruler vs the "
                 f"margins vs T(t) through g1c W1's flat phase -> {verdict}",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png1 = rd / "e242_wall_commitment.png"
    fig.savefig(png1, dpi=130)
    plt.close(fig)

    # ---- second figure: the thermal detail + the margin distributions
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 7.2))

    # (a) the per-probe margin distribution evolution
    ax = axes[0]
    data, labels = [], []
    for key in ["t0"] + [f"w1+{s}" for s in all_steps]:
        data.append([p["margin_sigma"] for p in cells[key]["probes"]])
        labels.append(key)
    bp = ax.boxplot(data, showfliers=True, patch_artist=True,
                    flierprops={"markersize": 2.5, "alpha": 0.5},
                    medianprops={"color": "k"})
    for patch, k in zip(bp["boxes"], labels):
        patch.set_facecolor("tab:red" if k == "t0" else "tab:purple")
        patch.set_alpha(0.45)
    ax.axhline(T204_FLIP_SIGMA, ls="--", lw=1.4, color="crimson")
    ax.annotate("T204 flip zone (0.05 sigma)", (0.02, T204_FLIP_SIGMA),
                xycoords=("axes fraction", "data"), fontsize=7.5,
                color="crimson", va="bottom")
    ax.set_xticklabels(labels, fontsize=7.5, rotation=20)
    ax.set_ylabel("per-probe argmax margin (sigma)")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(a) the battery's margin DISTRIBUTION through the flat "
                 "phase", fontsize=9.5)

    # (b) R2 + echoes per state
    ax = axes[1]
    thr = thermal_rows
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(np.arange(len(thr)), [r["R2_pooled"] for r in thr], "o-",
            ms=8, lw=1.8, color="tab:purple", label="R2 pooled (the one-T family)")
    ax.plot(np.arange(len(thr)), [r["R2_two_moment"] for r in thr], "^--",
            ms=6, lw=1.2, color="gray", alpha=0.8,
            label="R2 two-moment desk bound")
    ax.plot(np.arange(len(thr)), [r["T_lsq"] for r in thr], "v:", ms=5,
            lw=1.0, color="tab:olive", alpha=0.8, label="T LSQ echo (right axis)")
    ax.plot(np.arange(len(thr)), [r["T_kl_full"] for r in thr], "d-.", ms=5,
            lw=1.0, color="tab:cyan", alpha=0.9, label="T full-vocab KL echo (right)")
    ax.set_xticks(np.arange(len(thr)))
    ax.set_xticklabels([r["state"] for r in thr], fontsize=7.5, rotation=20)
    ax.set_ylabel("R2 decline-variance explained (left) / T (right)")
    axr = ax.twinx()
    axr.set_ylim(0.9, 1.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, loc="best")
    ax.set_title("(b) R2 + the echoes — the wall's p-side has little decline "
                 "to explain (SS flagged where degenerate)", fontsize=9.0)

    # (c) the verdict + the state table + the thermal quote
    ax = axes[2]
    ax.axis("off")
    y = 0.97
    ax.text(0.03, y, f"E242 VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.052
    for wd in textwrap.wrap(clause, width=62, break_long_words=False)[:8]:
        ax.text(0.03, y, wd, fontsize=7.0, va="top", family="monospace")
        y -= 0.023
    y -= 0.015
    ax.text(0.03, y, "state   ruler   med(s)   p25(s)  argZ  T_MLE   R2",
            fontsize=7.4, va="top", family="monospace", weight="bold")
    y -= 0.024
    for r, th in zip(traj, thermal_rows):
        ax.text(0.03, y,
                f"{r['state']:>7} {r['gm12_ruler']:7.4f} "
                f"{r['margin_median_sigma']:8.4f} {r['margin_p25_sigma']:8.4f} "
                f"{r['frac_argmax_z']:5.2f} {th['T_mle']:6.4f} "
                f"{th['R2_pooled']:+7.4f}",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.022
    y -= 0.012
    ax.text(0.03, y, "e238 124M w1 quote: T 1.060/+10 1.349/+50 1.453/+80; "
            "R2 0.299 0.626 0.690", fontsize=7.2, va="top",
            family="monospace", color="dimgray")
    y -= 0.03
    ax.text(0.03, y, "GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}" for g, v in gates_summary.items()),
        fontsize=7.0, va="top", family="monospace")

    fig.suptitle("E242 — the commitment layer's detail: margin distributions, "
                 "R2/echoes, the verdict table", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png2 = rd / "e242_thermal_leg.png"
    fig.savefig(png2, dpi=130)
    plt.close(fig)

    # ================= P5: honesty + provenance + close =======================
    metrics["honesty"] = {
        "e208_distinction": "e208's margin object was the FACT-EDGE over the "
                            "wash band (an instrument that died its honest "
                            "scope death); THIS cell's object is the "
                            "NEXT-TOKEN ARGMAX MARGIN in sigma vs arithmetic "
                            "noise — a different ruler that shares only the "
                            "word 'margin'; not a resurrection (T207/W031's "
                            "convention, restated from e229)",
        "n_and_scope": "ONE lineage (g1c's fresh root, ONE wall commit R=0.7 "
                        "raw, ONE wash draw seed 10902 — g2e's held-seed "
                        "convention), ONE battery (install-60 g-12, 60 "
                        "probes), n=1 deterministic reads per state (T204); "
                        "the other wall lineages (g1b/g1bR/g1d/g1e/g1f, the "
                        "10M take5/take6) are NOT read here — e229's t=0 "
                        "snapshot covers the cross-lineage axis at t=0",
        "thermal_leg_is_a_co_report": "NO BAR adjudicates the thermal leg "
                                      "(the dispatch's letter); T218's "
                                      "prediction (a) is discharged as a "
                                      "measurement: the wall's batteries DO "
                                      "admit a T(t) fit — the trajectory and "
                                      "R2 are quoted against e238's 124M "
                                      "unwalled numbers, both verbatim",
        "degenerate_denominator_flag": "the wall's p-side is nearly flat "
                                       "(the ruler moves < 5% through "
                                       "+300), so the R2 decline denominator "
                                       "is small; every row carries ss_tot + "
                                       "the degenerate flag — R2 values on "
                                       "degenerate rows are texture, not "
                                       "evidence",
        "determinism": "margins and logits are deterministic in-session "
                       "(T204: same code path, same device -> bit-exact); "
                       "the eval batch shape (1 x L, single-probe forwards) "
                       "is disclosed — batch-shape re-rounding sits at the "
                       "~3e-7 texture floor, far under every aggregate "
                       "here; the frac below T204's 0.05 sigma flip zone is "
                       "co-reported at every state",
        "ruler_growth_reads_as_flat": f"the +2 state EXCEEDS the root on the "
                                       f"committed trace (0.9094 > 0.9026) "
                                       f"and +300 exceeds it too "
                                       f"({gm12_300:.4f} vs {root_gm12:.4f}): "
                                       f"'decline' can be negative; the "
                                       f"clause fixes read growth as flat "
                                       f"(registered before compute)",
        "nothing_guaranteed": "the trajectories could have landed anywhere; "
                              "the observed outcome is recorded verbatim "
                              "against the frozen bars; no bar shopping",
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
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
                              {k: v["G"]["body_flat_md5"]
                               for k, v in cells_light.items()
                               if k != "t0"}},
        },
        "machinery": {
            "margin_instrument": "lab/e228_margin_landscape.py margin_pass "
                                 "MODULE-IMPORTED via e229's _E228NetShim "
                                 "(adapter only; arithmetic untouched): "
                                 "margin_sigma = (top1-top2 logit)/"
                                 "std(vocab logits) at the answer position, "
                                 "torch std unbiased",
            "thermal_instrument": "lab/e238_temperature_null.py TempFamily + "
                                  "TwoMomentFamily PORTED VERBATIM (copied, "
                                  "noted inline; grid 1201 + golden polish + "
                                  "NLL/R2 definitions byte-identical) — "
                                  "importing e238 would drag the 124M "
                                  "organism's transformers machinery into a "
                                  "2.74M CPU cell",
            "state_loads": "t0 via e225.load_body (the roster's own "
                           "convention); W1 states via g1's evl_load "
                           "(settle onto the ball + disarm — the lineage's "
                           "registered instrument adaptation), certified "
                           "per-state against g1c's committed W1 light-dial "
                           "record at tol 5e-3",
            "battery": "install-60 g-12 ruler (SPLICE_RNG 24301; 60 windows; "
                       "e225's module constants; e229's P0a ported; NO 124M "
                       "battery computed)",
        },
        "eval": {"device": "cpu fp32", "threads": torch.get_num_threads(),
                 "batch_shape": "1 x L per probe (e228's margin_pass shape)",
                 "n_forwards": f"9 states x 60 probes x 2 passes (margin + "
                               f"logit) = {9 * 60 * 2}"},
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(rd / "metrics.json"), str(png1), str(png2)]
    write_partial("P5 DONE (honesty + provenance + figures)")
    log(f"outputs: {rd / 'metrics.json'}, {png1}, {png2}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
