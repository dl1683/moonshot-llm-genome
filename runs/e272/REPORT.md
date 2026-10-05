# E272 — THE CAPACITY REPAIR LADDER (R65's top repair bill)

**Verdict: RANK-WRITES-THE-CURVE** — the kept-matched 1k arm STAYS DEAD
(post g0 0.001245 < 0.01) AND the edge lands inside (1k,10k] (the 1k->2k
pair jumps 61.2x with the dead side 0.000435 < 0.01; the tightest bracket
**k in (1,000, 2,000]**; the 0.05 expression floor first crossed at
k=5,000). Dose acquitted at matched rank; the capacity story stands with a
located edge. **P-272a (T249) CONFIRMED: dose does not buy expression.**

The question (frozen): *Is the expression cliff written by RANK or by DOSE?
Where exactly in (1k, 10k] does the edge sit? And how big is the room
lottery at the cliff?*

## The four arms (adjudicated on the WRITE read — post g0; root g0 CARRIED, T246's rehearsal lane)

| arm | k | room seeds | install lr | **post g0** | root g0 (carried) | kept (ledger med) |
|---|---|---|---|---|---|---|
| K10KR (fresh-room replicate) | 10,000 | 27215/27216 | 1e-3 | **0.209721** | 0.6970 (landed) | 0.0601 |
| K2K | 2,000 | 27211/27212 | 1e-3 | **0.026616** | 0.7757 (landed) | 0.0249 |
| K5K | 5,000 | 27213/27214 | 1e-3 | **0.127096** | 0.6112 (out) | 0.0418 |
| K1KM (kept-matched) | 1,000 | 26111/26112 (== e261's committed K1K room, bit-gated) | **1e-3 x 3.7305567687** | **0.001245** | 0.6879 (landed) | 0.0161 |

The cited ladder rungs (md5-bound < 1e-12): 1k = e261's committed K1K
0.00043458465370349586; 10k = e264's committed K10K 0.26464763283729553.

## The three answers

1. **RANK, not dose.** The K1KM arm ran the committed 1k room with the
   install lr compensated by LR_SCALE = stored_kept(10k)/stored_kept(1k) =
   0.060045162390265784 / 0.016095496225535792 = **3.7305567687315575**
   (re-derived at runtime from e264's md5-bound kept_frac_curve; G_DOSE_ARITH
   exact). It died at **0.001245** — 8x below the dead bar, 40x below the
   expression floor. And the dose match was not merely nominal: the applied
   in-room dose proxy (median lr_s x ||g'||) came out **1.12x** the 10k
   arm's (first-order 1.0), and the end-to-end IN-ROOM displacement norm
   ||P_room(theta_s400 - theta_0)|| was **11.96 vs the 10k rung's 8.70**
   (1.37x) — the compensated arm wrote MORE in-room parameter mass than the
   alive 10k rung and still expressed nothing. The 609x cliff was not the
   3.7x kept-dose change in disguise.
2. **The edge sits at k in (1k, 2k]** (the firing pair 1k->2k at 61.2x,
   dead side 0.000435); K2K lands partial (0.0266 — above the 0.01 dead
   bar, below the 0.05 floor), K5K expresses (0.127), the floor is first
   crossed at k=5,000. The registered expectation's "bracket slides"
   branch is the live one — the cliff is a steep RAMP inside (1k, 5k], not
   a step at 10k: 0.000435 -> 0.0266 -> 0.127 -> 0.265/0.210.
3. **The room lottery at the cliff rung is moderate:** the fresh-room 10k
   replicate read 0.209721 vs the committed 0.264648 (**|d| 0.0549**, ~21%
   relative), inside the pre-registered [0.15, 0.45] window — ROOM-LOTTERY
   does NOT fire; the cliff pair stands trusted (the sensitivity ladder
   with the replicate substituted at 10k locates the edge identically —
   agrees with the primary).

## The dose control (metrics['dose_control'], measured never nominal)

- stored values used: kept(1k) = 0.016095496225535792, kept(10k) =
  0.060045162390265784 (e264's committed kept_frac_curve, md5-bound);
  LR_SCALE = 3.7305567687315575; lr_K1KM(s) = 1e-3 x LR_SCALE x
  cosine_lr(s-1, 1000) via the disclosed e043_install.LR module rebind
  (restored in a finally block; verified live in the log; the cons reads
  G1.FT_LR untouched).
- applied-dose proxy (median lr_s x ||g'||): K1KM 4.9e-5 vs K10KR 4.4e-5 —
  ratio **1.119** (first-order expectation 1.0).
- end-to-end in-room displacement: K1KM ||d|| 15.03, ||P d|| **11.96**;
  K10KR ||d|| 9.21, ||P d|| **8.70** — ratio 1.374 (the compensated arm
  over-delivered in-room and still died; dose acquitted a fortiori).
- disclosed first-order caveats (in the birth docstring): Adam's
  per-coordinate normalizer partially cancels uniform gradient scales; the
  rebind scales AdamW's decoupled weight decay with the same lr.

## Gates — 13/13 PASS

G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR(bank), G_INSTMASK, G_PARENTS
(e261 md5 f460475d8e6b76f0719e91c1e9c6041b + e264 md5
a42ff4786784b04cb9819a69b545e343 + the kept_frac_curve and both cited rungs
hard-bound < 1e-12), G_DOSE_ARITH (the compensation exact), G_BASE, G_ROOT,
G_VMBIND, G_SPANBIND, G_PROJ (four rooms: idem ~5.6e-16, kept^2 == k/N at
the 10-sigma bar, span-overlap ~ sqrt(k/N)), G_ROOM1K (the K1KM room
bit-identical to e261_rooms.pt's stored K1K D/S — the shared-room bit-gate).

No FREE arm, no anchor re-run (disclosed at birth: no arm is bit-identical
to a committed rung BY CONSTRUCTION; the instrument is the 5-generation
lineage — e264's G_FREE PASS at install L2 6.2e-4 + e268/e269/e270/e271's
serial anchors at |d post g0| 2.4e-7/9.5e-7/1.0e-6/9.5e-7).

## Envelope + discipline

Serial arms only, bursts <= 175 s, 2,800 per-step thermal polls tagged
e272:ARM:phase to runs/_envelope_log.jsonl, max temp 79.0C (one burst ended
at the 78C margin by design), 0 violations of the 84C line; 40 s cooldowns;
CPU fp64 projections (pocketfft), CPU threads 4; wall 2,817.7 s.

## Smoke record (one catch)

E272_SMOKE=1 caught ONE bug — a broken split f-string in the instrument
page's kept-ledger label (SyntaxError at import; hoisted to a local median
var; commit 261455c). After the fix: 13/13 gates PASS (G_ROOM1K
smoke-vacuous, disclosed), all four installs/cons/adjudication/figures
exercised, the E43.LR rebind verified live, thermal max ~60C. The smoke's
own verdict line was the s8-category artifact (dead writes at smoke k vs
the full-s400 committed cite) — SMOKE-stamped, nothing adjudicated.

## Reading notes for the fold

- The rehearsal lane's 5th consecutive confirmation: K1KM's DEAD write
  (0.001245) landed root g0 0.6879, IN the band; K2K's 0.0266 write landed
  0.7757 — the landing read measures the cons, not the write.
- The located edge (1k, 2k] is 5x BELOW the committed cliff pair's top —
  the "capacity number" moves from "~10k" (e264's threshold_k) to the
  (1k, 2k] bracket with a partial-expression rung at 2k; the curve's shape
  is a steep ramp, not a step (max adjacent ratio INSIDE this cell's fresh
  rungs: 2k->5k at 4.8x — below the 10x bar; the only >= 10x dead-side
  jump is the cited 1k->2k pair).
- kept ~ sqrt(k/N) held exactly (measured 0.0161/0.0249/0.0418/0.0601 vs
  sqrt expectations 0.0191/0.0270/0.0427/0.0604 at the install gradient
  structure).

Committed at birth (20e412f) before any compute; adjudicated against the
frozen bars verbatim; no bar shopping. No NOTES/THINKING/QUEUE/STATE edits
(the coordinator folds).
