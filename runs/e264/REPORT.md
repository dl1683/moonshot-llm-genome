# E264 — THE MIDDLE RUNGS (recovery cell): the cliff pinned — SHARP-THRESHOLD at k ~ 10,000

Cell: e264 (e261's deferred ladder, recovered; T239's registered ask).
Date: 2026-10-05. Script: `lab/e264_middle_rungs.py` (committed at birth
a789e10 BEFORE any compute; pass-2 amendment bed2114 and pass-3 bookkeeping
fix dc8ff4c, both disclosed pre-compute). This report + the draft NOTES
entry are the deliverables; the coordinator folds.

## THE VERDICT (adjudicated against e261's ORIGINAL frozen bars, verbatim)

**SHARP-THRESHOLD** — "the expression curve is step-like (post_g0 jumps
609.0x between k=1000 and k=10000; the floor 0.05 first crossed at
k=10000) AND the landing curve enters the band at a locatable rung (first
in-band k=10000, bracket 1000->10000) — the anti-substrate's final form:
a dimensional threshold at k ~ 10000; memories need k* dimensions, full
stop."

All 14 gates PASS (incl. G_FREE tiers i-iii and the K10K resume bit-gate);
composite order TEXTURE -> SHARP -> GRADUAL -> MIXED resolved at SHARP.

## THE FULL 5-RUNG CURVE (the completed ladder)

| k | post g0 (expression) | root g0 (landing) | kept | in-band | source |
|---|---|---|---|---|---|
| 10 (context, not a rung) | 2.86e-5 | — | — | dead | e246 committed |
| 1,000 | 0.000435 | 0.5769 | 0.0161 | under | **e261 committed** (cited, md5-bound) |
| 10,000 | **0.264648** | **0.7708** | 0.0600 | **IN-BAND (the threshold rung)** | e264 (the K10K resume) |
| 40,000 | 0.346476 | 0.7520 | 0.1211 | IN-BAND | e264 fresh |
| 100,000 | 0.435982 | 0.7276 | 0.1915 | IN-BAND | e264 fresh |
| 237,123 | 0.384364 | 0.7106 | 0.2946 | IN-BAND (scatter-fragile) | **e261 committed** (cited) |
| FREE (k=N) | 0.5302 | 0.7156 | 1.0 | the ceiling | e264 fresh (e261's FREE 0.7691; g1c committed 0.7448) |

- Adjacent expression ratios: **1k->10k 608.97x (the >10x bar FIRES
  here)**; 10k->40k 1.31x; 40k->100k 1.26x; 100k->237k 0.88x. The entire
  cliff lives in ONE decade of the rank axis; above it the expression
  curve is flat-ish (0.26-0.44, mild rise then a within-lottery dip) and
  the landing saturates in-band.
- The landing curve above the cliff reads FLAT within the cons scatter
  (0.7708 / 0.7520 / 0.7276 / 0.7106 across four decades of k) — the
  slight non-monotonicity is inside the lottery's measured range (below).
- The kept covariate tracks sqrt(k/N) everywhere (0.060 at the threshold
  rung = a SIX-PERCENT-dose write). The highest root g0 of the whole
  ladder belongs to the SMALLEST living room: dose does not write the
  curve's shape; rank does (e258's VLIGHT kept-0.10 landing, now
  triple-confirmed).

**The cliff's location: k* is inside (1k, 10k] — between 0.036% and
0.365% of the 2,739,072-dimensional parameter space.**

## THE STITCH + THE CONS-SCATTER DISCLOSURE (carried, as dispatched)

- Rungs 1k/237k are e261's committed reads (metrics md5 f460475d,
  hard-bound, values asserted < 1e-12). Post-install reads stitch cleanly
  across sessions: the install is deterministic (e261's G_ANCHOR: install
  L2 6.7e-5, |d post g0| 5e-7 on a bit-identical arm; this session's FREE
  reproduced e261's FREE install L2 6.2e-4, behavioral |d| 1.3e-5).
- Root reads carry the cons lottery's cross-session scatter (e261's
  G_ANCHOR: |d root g0| 0.0419, |d root g-12| 0.1483). Per-rung
  scatter-fragile flags: ONLY k=237123 (0.7106 sits 0.0403 above the
  band floor, inside the 0.0419 reach — the named instance where e260's
  committed 0.6686 read OUT by 0.0017). The threshold rung's calls are
  clear of the scatter (10k: 0.1005 from the floor, 0.0484 from the
  ceiling).
- **THE CONS-SEED REPLICATE (the anchor scatter priced, T239's ask):**
  e261's committed K237K install-final + a fresh cons at seed 10902
  (the next seed; nothing shopped) -> root g0 **0.7243**: |d| 0.0137 vs
  the cited rung (0.7106), |d| 0.0557 vs e260's committed (0.6686);
  root g-12 0.8793 (|d| 0.5220!); displacement in-own-room 0.6674; the
  237k room reconstructed bit-identical to e260_rooms.pt (D/S exact).
- **Three cons draws on the same bit-identical 237k arm:** root g0
  {0.6686, 0.7106, 0.7243} — range 0.056, all in-or-near-band; root g-12
  {0.209, 0.357, 0.879} — range 0.67, a WILD lottery. The g0 landing is
  the stable ruler; the g-12 read at root is a coarse draw (the
  adjudication never used g-12). The FREE ceiling scatters the same way
  across sessions: {0.7448, 0.7691, 0.7156}.
- e261's G_ANCHOR failure is inherited as this stitch's disclosed noise
  floor, NOT as an e264 gate (no anchor rung is re-run here; no bar
  moved — frozen in the registration).

## THE GATES (all PASS)

G_NAMEFREE, G_SPLICE (19+41), G_BATTERY, G_ANCHOR(bank), G_INSTMASK,
G_PARENTS (g1c/e246/e258/e260 + the e261 stitch bind), **G_K10KRESUME**
(the resume vehicle bit-gated BEFORE compute — pass 1 bound e261's cut at
md5 ce70e580/s278 and continued it; pass 2+ bound the completed vehicle
at md5 0f6dc1cf/s400, traj [1,100,200,300,400], ledger to s400 — the
original cut's md5 preserved in the birth commit), G_BASE (e001
fact-free), G_ROOT (battery read == committed to 1e-10), G_VMBIND,
G_SPANBIND, G_PROJ (the three middle rooms certified: idem ~6e-16, kept2
== k/N at the 10-sigma bar, span-overlap == sqrt(k/N)), **G_FREE** (this
session's ceiling: install L2 6.24e-4, behavioral |d| 1.3e-5, root 0.7156
in band, cons tracking median 0.0022), G_FREE_XCHECK (co-report).

## THE K10K RESUME (the dispatch's check #1)

e261's cut vehicle (s278/400, seeds 26113/26114) was continued bit-exactly
by the journal-resume path (model + optimizer + generator state + ledger);
the resume is visible in runs/e264/run.log ("RESUMED ... at step 278/400",
traj continuing [300, 400]). The vehicle now holds the s400 final (the
path's design; the cut state preserved by the birth-commit md5).

## INCIDENTS + DEVIATIONS (all disclosed, all in metrics.json)

1. The predecessor executor died ~2 min in on a model-request failure;
   verified NO partial artifacts — this cell re-dispatched whole.
2. Pass 1 observed ONE pathological 29-minute training step (15:15-15:44Z;
   CPU-side projection stall under transient background load; GPU stayed
   cool); pace recovered fully on its own.
3. Pass 1 was killed mid-K100K-install BEFORE that arm's first chunk-end
   save (nothing lost — the arm restarted fresh in pass 2 and reproduced
   pass 1's partial trajectory exactly: s100 g0 0.4980 in both).
4. Pass 3 = record hygiene only (one bookkeeping line: the G_K10KRESUME
   post-read compared against a stale pass-1 expectation; no data or
   adjudication touched).
5. n=1 per rung, one lineage, TWO sessions stitched; the curve's SHAPE is
   the registered object. No washes (inherited); the v-map LOADED
   (extend, don't repeat); kept-k coupling disclosed and co-plotted.

## THE THERMAL ENVELOPE

2,964 envelope polls tagged e264 (the owner's max-priority window):
**max 78.0C (the burst-end margin itself), zero reads >= 84C**, bursts
<= 175s, 40s cooldowns, per-step polls. (The final metrics.json's
thermal block shows pass 3's zero-GPU fast-path; the envelope log +
run.log chunk tables carry the real compute's record.)

## WHAT THIS COMPLETES (for the fold)

The anti-substrate's complete arc: e246 (the corpus's directions cannot
write facts; geometry irrelevant) -> e258 (not the load) -> e260 (rank is
the barrier) -> e261 (the coarse cliff inside [1k, 237k]) -> **e264 (the
cliff pinned: k* in (1k, 10k], ~0.04-0.37% of dimensions; expression
cheap and the landing saturated at matched strength from the first living
rung)**. T239's "memory capacity's geometric form" now has its number's
bracket. Still open elsewhere: the fine bracket inside (1k, 10k] (a
{2k, 5k} pair would finish it); T237's occupancy question (does the
corpus's own room erode differently — the anti-substrate's only surviving
branch); the g-12 root lottery (three draws, range 0.67) as an
instrument note.

Artifacts: runs/e264/{metrics.json, e264_middle_rungs.png,
e264_instrument.png, REPORT.md, DRAFT_NOTES_ENTRY.md, run.log
(gitignored; three passes appended)}; checkpoints runs/checkpoints/
e264_* (rooms, 4 roots, the replicate root, resume vehicles).
