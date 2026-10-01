# R60 AUDITOR — the R59-inherited recomputation mandate, discharged (2026-10-01)

Mandate: R59's auditor died pre-report; its recomputation duty transfers here,
plus this wave (e191, e192, opt1b2, opt1b3, e_chart, e182c, x1, g1bS, g2g;
T144-T152; the W024 retirement; T139's re-amendment; R59; the ledgers).
Every headline number below was recomputed from `runs/*/metrics.json` (and
git history where the live file was superseded). The R60 critic's repairs
(commit 51bb934) landed DURING this audit; findings that converge with them
are marked [CONVERGES-R60c].

## 1. The terrain chain

### e191 (STATIC-CLIFF) — SOUND, digit-exact
All quoted profile values reproduce from `runs/e191/metrics.json`:
pump 0.955532 @ D 0.33; peak 0.960461 @ 0.20; 0.910017 @ 0.50; 0.755210 @
0.66; 0.491560 @ 0.80; DEAD 0.248182 @ 0.92; floor 0.000434 @ 2.0; edge
bracket [0.80, 0.92]; CE_R 3.0035 at the kill / 5.2806 at D 2.0 (fact canary
confirmed); 5-point crosscheck max |d| 1.6e-6; dyn kill interp 0.92034.
STATIC-SPARES correctly does not fire (0.2482 < 0.27). Nothing to repair.

### e192 (TERRAIN-ONE-PICTURE + RIDER) — SOUND, digit-exact
Five-ray table reproduces: g-ray first-dead D 0.92 (RMS 5.559e-4); static
sign(g0) 2.5 (RMS 1.511e-3; 2.5/0.92 = 2.72x, kill bracket [2.0, 2.5] inside
the registered [2.0, 3.2]); three Gaussians alive-flat at D 4.0 — 0.904616 /
0.897322 / 0.895470 (NOTES "0.896-0.905" correct; > 4.35x the g-ray).
Pumps: g +0.044872, sign +0.037254, Gaussians +0.000169 / -0.0000557 /
-0.0000848 (no ridge anywhere; the 0.02 floor holds). A0 stitch gate:
0.678596 vs committed 0.678044, d 5.52e-4 — "ON the static curve" verified.
Rider: step_L2 0.0055143 = 1.6543/300; alive 0.433994 @ D 0.827127 → dead
0.135193 @ D 0.992570 (bracket [0.827, 0.993] correct; RIDER-STEPSIZE-OWNS
dead 0.000728 @ 1.654253). G_RIDER_XC anchor d_gm12 4.36e-8 ("4.4e-8"
correct). The fresh g-ray is bit-exact against e191's profile on all 12
shared Ds (max |d| 0.0 — recomputed here directly). Two phrasing notes, not
errors: (a) T146's "isotropic inert (+0.0002)" quotes the single most
positive Gaussian of three (the others are negative) — the fired bar is "no
pump ridge," which holds 3/3; (b) the e191 static ray being the same object
as the dynamic path inside [0, 1.6543] is disclosed in both files.

### e_chart (A2 + B2 + riders) — SOUND on every gate; four NOTES imprecisions
Hard-gated anchors: G_W024 same-point raw +0.098621 vs committed +0.098077
("d_raw 5.4e-4"), opt1-estimator -0.038508 vs committed -0.038513 (d 5.40e-6);
attenuation +0.0986 → +0.0396 = 2.49x ("~2.5x" correct). A2 classes
[+0.0538, +0.0831, +0.0402, +0.0266] all positive; top-0.1% ||g||^2 share
0.40066 ("40%" correct); 0 cuts in 21 states. B2: e131 in-span kill rungs
[1, 2, 1] (interp thresholds 1.0/1.5145/1.0), fine-D in-span 0.5637/0.6101
("0.56-0.61" correct); out-span removes 0.642112 of the ray, residual L2
0.76675; store/host removed 1.000001/0.999999 ("removed 1.0" correct),
out-span kill rung 8 both (interp 6.26/4.14). Riders: shuffled sign 0.916008
→ 0.930556 through D 2.5, CE_R flat 1.663, total rise +0.0146 < the declared
0.02 floor — INERT confirmed; wider grid: R3 alive 0.792275 at D 12.0
(12/0.92 = 13.04x, ">12, >13x" correct), with the D 4.0 anchor bit-equal
(d 0.0).
NOTES-side imprecisions (repairs R1a-R1d below):
(a) "kappa-derived d_eff >=52k/36k/24k" — the inequality is BACKWARDS and
one value mis-rounds. The kappa source is "unresolved-high ... bounded BELOW
by 12/wash-1x rungs", so d/kappa^2 is bounded ABOVE: **<= 52,055**; store is
34,988 ("35k", not "36k"); host 24,278. [CONVERGES-R60c: the critic already
corrected the claims ledger to "<=52k"; NOTES.md still carries ">=52k/36k/24k".]
(b) "PR 6.4-15.8" is e131-only (its three PR variants 6.41/11.47/15.81);
store 5.22 and host 4.76 sit below that range — the line reads as if it
spanned all three organisms.
(c) "the out-span residual retains 59% of the g-ray" is the L2-squared
(energy) share (0.76675^2 = 0.588); the metrics' own note carries the L2
share 0.7667 — say which.
(d) "in-span random kills at rung 1 on all three organisms" suppresses e131
draw 11602 (seed), which killed at rung 2 / interp 1.51 — a 3x threshold
spread the adjudication clause itself carries. [CONVERGES-R60c: the critic
restored this disclosure on the ledger; NOTES still compresses it.]

### opt2 (the density ladder) — SOUND on all five rungs; one UNSUPPORTED number
Ladder recomputed: topk-10 0.906645; topk-50 0.920260; raw (opt1c committed)
0.920341; sign 1.749640; A0-Adam (opt1 committed) 2.489262. TOPK-50 == raw
at 4 digits (they differ at the 5th: 0.92026 vs 0.92034 — "four-digit echo"
is exactly right). GRADED correct (neither named bar; all arms kill; DENSITY-
CARRIES required a_sign to SPARE past 2.5 — it killed at 1.75). G_SAMEPOINT
both anchors d 0.0. The canned stretch phrase ("the sign kill MOVES with
horizon") does read against the number (the kill arrived 30% BELOW the static
edge — more lethal, not horizon-stretched); the fold carries the numbers,
not the phrase, in both NOTES and T151. Honest.
**FINDING (repair R2): NOTES's "the top-10% front holds 75% of ||g||^2" is
backed by NO artifact** — absent from runs/opt2/metrics.json, the run log,
and lab/opt2_density.py; the only committed census of this quantity (e_chart,
same t=0 state, same batch) reads share_top(k_frac=0.1) = 0.8660. Either
correct to 86.6% (e_chart's census) or delete the parenthetical. The
adjacent "reads ~0.5 alive there" is also loose: the static sign ray at the
sign-path kill D 1.75 interpolates to ~0.61 (0.764 @ 1.5 → 0.448 @ 2.0);
at 2.0 it reads 0.448. Quote 0.45-0.61 or the grid values.

## 2. The corrections: is attenuation consistent EVERYWHERE? — NO, three survivors
- T139: correct in place (R58 flip amendment + the CHART RE-AMENDMENT
  bracket, attenuation form). OK.
- scratch/claims_ledger.md abstract: "attenuates the stream's fact-relevance
  ~2.5x at a matched point" — correct (and further hardened by the R60
  critic). OK.
- DAY_SEVEN_REPORT.md: "dissolved into an estimator artifact" — correct. OK.
- **scratch/day6_paper_skeleton.md R2b (~line 114): STILL SAYS "the
  normalizer flips the sign of fact-relevance, -0.0385 vs +0.0981"** — the
  paper's own wash-law clause, exactly the location the mandate names.
  REPAIR R3a.
- **scratch/day7_skeleton.md section 3 (~line 26): STILL SAYS "The
  normalizer FLIPS THE SIGN of fact-relevance (-0.0385 vs +0.0981, same
  batch)"**; the same section still carries W024's retired stitches/cuts
  picture ("the pump is a few big stitches; the erosion a thousand tiny cuts
  — the healing signal is OUTVOTED") with no retirement marker. REPAIR R3b.
- **THINKING.md T141 (~line 995): "e189's census survives as the mechanism
  of the FLIP (W024), not the currency"** — an uncorrected flip reference;
  the census survived as the ATTENUATION/estimator-point mechanism, and there
  is no flip. One bracketed amendment line owed. REPAIR R3c.
(REVIEWS R58's critic text and W024's pre-mortem body are historical record
and correctly left as-is; W024's header carries the RETIRED-BY-CHART stamp,
and T131 carries the G2G RESOLUTION — both verified.)

## 3. The negative cells

### g1bS — SOUND; the hard stop is exactly what the metrics show
G-BASE-QUAL: g1 cosine-complete PASS (4000/4000, 13 chunks); g2 FAIL —
chunk-final vals 2.16088 → 1.57657 (s1113 min) → 2.85057, first violation
chunk 5 (1.59959 > 1.57657+0.02) exactly as NOTES says; g3 FAIL (2.8506 >
1.70); g4 PASS (ls 0.7133, mwl 4.25, max run 2, 42 distinct). `pass: false,
enforced: true`; the metrics contain NO arms/adjudication keys; the generated
sample is recorded verbatim (the memorization-recitation exhibit). Archives
on disk: g1bS_base_diverged_s3250.pt and g1bS_base_overtrained_s4000_lr4e-4.pt
— never deleted. The "1.568" 1e-3 minimum IS artifact-backed: committed in
the b7f8e20 survivor partial (`recovery.third_dispatch.divergence_verification
.val_min` = 1.567773 on the s3250 diverged ckpt), with the pre-min trajectory
(s353 1.8385 → s750 1.5801 falling) bracketing the U-turn window; 4e-4 min
1.576568 = the final metrics' chunk 4. "Both lrs U-turn in the same s~1000-
1200 window" is supported (4e-4 min at s1113; 1e-3 min bracketed (750, 1207)).
The x1 skip of the vanished s1113 state is documented in runs/x1's honesty
block. Model negative cell; nothing to repair.

### opt1b3 — SOUND; every quoted number recomputes
150 every-step reads: min 0.289149 AT s1350 (the final step); mean 0.316694;
no read <= 0.27; final D 2.250087; D(t) slope 6.973e-4/step, r2 0.999633;
mean step_disp 0.006443 ("walked 6.4e-3"); sublinearity 9.2604x; erosion
linear-fit -2.03e-4/step (0.3296 → 0.2891); margin over the 0.27 bar 0.0191;
CE_R 1.6771 (root 1.6635); cos(g0) -0.0192..-0.0199. CAP-AGAIN graded
correctly (frozen composite order); the third falsified projection stated
with its numbers; the extrapolation labeled non-adjudicating. The fold
carries numbers, not phrases. Nothing to repair.

### e182c — SOUND; the replay premise is disclosed AND gated
Premise correction disclosed pre-compute (NOTES) and enforced by gates:
G_REPRO_CORPUS filter equality (40001 lines, 664 dropped, 1,093,972 chars,
331,770 tokens — both sides identical); G_REPLAY bit-tight vs e182's record
(dp <= 0.0010593 at +50, ppl ratios <= 1.00034). Declines: +80 fact 0.4294
vs ctrl 0.39695 (ratio 0.9244); +50 0.33936 vs 0.28140 (0.8294) — T149's
"0.83" correct. Near-related retention at +80: 1 - 0.76554 = 0.23446 ✓.
Bank ppl 71.3357 → 34.8189 ✓ (improving while probes erode). FORGETTING-
GENERIC fires inside the registered 1.5x band; the GPT-2 clause rewrites
landed where promised: day6 gap-13 rewritten to "ordinary forgetting with
improving perplexity", and T123 carries the E182C RESOLUTION block. One
placement quirk: the resolution block sits after the e187 card's tail
(between cards) rather than inside T123's body — cosmetic; a pointer line
inside T123 would help future greps. Nothing substantive to repair.

## 4. g2g — SOUND on all five bars, both runs; the fragility is the best-folded part
Final metrics = run 2 (the rider-repair rerun) with run 1 embedded in
cross_run. Ladder rates [0.0333, 0.0333, 0.0400, 0.0400] identical both
runs (10/10/12/12 events; "10 -> 12 across 8x threat" correct; monotone, 2
distinct); L1 bit-reproduces g2c 10/10 both runs (0.6188/0.6186). Ceiling:
L4 frac-at-floor 1.0, cycle-median 0.0017/0.0016, duty 0.0. REFRACTORY:
R8 min spacing 8, 5/14 below 20 (floor frac 0.1429 + out-of-band 0.2143),
9/14 in 20-45 (0.6429), cycle-median 0.6705/0.6704 > L1 0.6188/0.6186,
duty 0.7407 — "maintenance IMPROVES at the shorter refractory (0.671 vs
0.619; duty 0.74)" all verified. Head-to-head: 1x +0.0758 run 2 (run 1
+0.072 per its commit d071899) — "+0.072/+0.076 both runs" correct; 4x
+0.0004/+0.0004 ("both dead") correct. The float-fragile 2x leg: run 1 organ
0.2405 vs run 2 0.1674 (last event 296→292 flip under GPU float
nondeterminism), deltas +0.076 → +0.007 — cross_run names the cause, bars
the citation of the 2x "win", and the standing letter (FIXED-MATCHES-OR-WINS
with the 1x win co-reported) matches run 2's adjudication. The 0.693
co-read: T131's G2G RESOLUTION supersedes it as endpoint-vs-median, with
T130's R56 amendment preserving the history — chain intact. W023 rider:
L1 median jump 0.2096 ("+0.21 at 1x" correct); late-half steeper 7/10 (1x),
6/10 (0.5x), 7/11 (2x) — "60-70% of events at 0.5-2x" correct; L4 median
jump 0.0001 ("pinned ~0" correct). One phrasing note: "At 0.5x = 1x
exactly" is rate-only (maintenance medians differ: 0.5669 vs 0.6186) —
co-report the medians or say "rate = 1x exactly".

## 5. Ledger integrity
- QUEUE vs runs: every DONE row of the wave (e191, e192, e189+e190→e_chart,
  x1, opt2, opt1/opt1b/opt1b2/opt1b3, e188, e182c, g1bS, g2g) has a
  committed metrics dir; the working tree is clean apart from STATE.json /
  INBOX (trio in flight) and the r60 scratch files. **The e193 row is
  DUPLICATED verbatim (two identical rows)** — delete one. e193 DISPATCHED
  17:04Z is the live fleet cell; STATE.current_experiment still names the
  R60 trio + g1bS2 + the seed ladder but NOT e193 — update on the next
  heartbeat ride.
- STATE clock notes: the clock_note documents both jumps and correctly
  demotes stamps to "best-effort labels"; ordering + hashes carry the record
  — consistent with how this audit used b7f8e20. last_novelty (09-30 12:15Z)
  is ~29h stale against the 2h field; the T138 lit beat exists, so this is
  field drift again (R58 already re-stamped once) — re-stamp with a note or
  invoke the clock_note explicitly in the field.
- DAY_SEVEN_REPORT: every bullet traces to a verified artifact (0.6%/CV 0.56;
  0.92/2.5/>12/>13x; 0.92→1.75→2.49; 1683x; 0.33→0.29; rider; +0.099→+0.040;
  ratio 0.92; s~1000-1113; ~3x). One STALE LINE: "Open: the rhythm's
  controls (g2g, out)" contradicts the report's own header ("the g2g verdict
  landed and is folded into C7") — replace with the actually-open item (the
  seed-replicate ladder). "Sources ... T136-T151" should read T136-T152.

## Prioritized repairs (anchored)
- **R1 (NOTES e_chart entry, the dimension line):** ">=52k/36k/24k" →
  "<=52k / 35k / 24k" (kappa bounded below ⇒ d/kappa^2 bounded above);
  "PR 6.4-15.8" → "(e131 PR 6.4-15.8; store 5.2, host 4.8)"; "retains 59%"
  → "retains 59% of the ray's L2^2 (0.767 of its L2)"; restore the 11602
  rung-2 disclosure to "kills at rung 1 on all three organisms". [Converges
  with 51bb934's ledger repairs; NOTES is the uncorrected copy.]
- **R2 (NOTES opt2 entry):** delete/correct "the top-10% front holds 75% of
  ||g||^2" — no artifact anywhere carries 75%; the committed census says
  86.6% (e_chart share_top @ k_frac 0.1). Optionally tighten "~0.5 alive
  there" to the grid values (0.448 @ 2.0; ~0.61 interpolated @ 1.75).
- **R3 (the flip's surviving copies — the mandate's consistency question):**
  (a) day6_paper_skeleton R2b: "flips the sign" → "attenuates ~2.5x at a
  matched point (the trajectory-level negative read is the post-step view)";
  (b) day7_skeleton section 3: same correction + mark W024's stitches/cuts
  sentence RETIRED-BY-CHART; (c) THINKING T141 line ~995: bracket "the FLIP"
  with the chart's estimator-point resolution.
- **R4 (ledger hygiene):** QUEUE — delete the duplicated e193 row;
  DAY_SEVEN_REPORT — strike "(g2g, out)" from the Open list, fix the T-range
  to T152; STATE — name e193 in current_experiment on the next ride; re-stamp
  or annotate last_novelty.
- **R5 (optional wording):** g2g NOTES "At 0.5x = 1x exactly" → add "(rate;
  maintenance medians 0.567 vs 0.619)"; T146 "isotropic inert (+0.0002)" →
  "no pump ridge on any of the three rays (max +0.0002)".

## Bottom line
The wave's numbers are real: of ~60 recomputed headline figures, every one
reproduces from the committed metrics except ONE unsupported co-read
(opt2's "75% of ||g||^2", R2) and one inequality mis-written in NOTES
(the d_eff bound, R1a — already fixed on the ledger by the critic). The
negative cells (g1bS, opt1b3, e182c) are the wave's most honest artifacts.
The one systemic gap is exactly where the mandate pointed: the estimator
correction reached T139, the claims ledger, and the day-7 report, but two
paper skeletons and one THINKING line still mint the dead "flip" — the
corrections-cascade day has three uncorrected copies of its own correction.
