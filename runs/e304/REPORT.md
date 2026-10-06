# E304 — THE FEVER CELL — MIXED

**2026-10-06T18:14:51.825356+00:00** — eval-only CPU desk cell (threads 4, no GPU, no envelope
writes). The honesty audit of the founding success: the error-gated
controller's 3.5x read overshoot through the one-T thermal lens, at the
gate tokens vs the held-out paraphrase — does preservation buy
confidence the organism cannot cash?

## The verdict

**MIXED** — the founding state (e288 ERROR-GATED
post, T fit against the loaded fact — its verified reference anchor): gate T
1.8258 (heat +0.8258), paraphrase T
2.5998 (heat +1.5998);
FEVER requires gate heat >= +0.1 AND para heat < 0.5x gate heat: gate +0.8258, para +1.5998 -> para/gate = 1.937.

## (a) THE TWO-SITE T TABLE (the discriminator, verbatim)

| fam | arm | role | GATE T (heat) | gate-y129 T (heat) | PARA T (heat) | KL-state co-report (gate/para) |
|---|---|---|---|---|---|---|
| N | BASE(loaded-fact) | base | 1.0000 (+0.0000) | 1.0000 (+0.0000) | 1.0000 (+0.0000) | 1.0000 / 1.0000 |
| F | BASE(e291-organism) | base | 1.0000 (+0.0000) | 1.0000 (-0.0000) | 1.0000 (+0.0000) | 1.0000 / 1.0000 |
| N | ERROR-GATED | post | 1.8258 (+0.8258) | 2.8730 (+1.8730) | 2.5998 (+1.5998) | 1.2992 / 1.6869 |
| N | NAME-FIXED-TWIN | post | 1.1439 (+0.1439) | 1.9717 (+0.9717) | 1.9383 (+0.9383) | 1.0631 / 1.4252 |
| D | CONTRADICTED-WITH-CONTROLLER | post | 1.8969 (+0.8969) | 3.1091 (+2.1091) | 2.8293 (+1.8293) | 1.3358 / 1.8060 |
| D | CONTRADICTED-NO-CONTROLLER | post | 1.3037 (+0.3037) | 3.7328 (+2.7328) | 3.6139 (+2.6139) | 1.1031 / 1.5341 |
| D | C1:ERROR-GATED | post | 1.8116 (+0.8116) | 2.7731 (+1.7731) | 2.4992 (+1.4992) | 1.2716 / 1.6451 |
| D | C1:NAME-FIXED-TWIN | post | 1.2050 (+0.2050) | 2.1786 (+1.1786) | 2.1493 (+1.1493) | 1.0798 / 1.4562 |
| F | FIVE-CONTROLLERS | post | 1.0902 (+0.0902) | 1.7808 (+0.7808) | 1.9222 (+0.9222) | 0.9744 / 1.1496 |
| F | SINGLE-CONTROL-TWIN | post | 1.2564 (+0.2564) | 2.0210 (+1.0210) | 2.1478 (+1.1478) | 1.0779 / 1.2531 |
| X | SANCTUARY-TWIN(e287) | post | 1.7226 (+0.7226) | 3.9176 (+2.9176) | 3.8221 (+2.8221) | 1.1951 / 1.4956 |

The y129 column is the battery read position alone — the site where the
committed 3.5x overshoot is defined. The KL co-report is
argmin KL(softmax(L_state/T) || softmax(L_ref)) — the shape-residual
form (the mass-vs-spike decomposition's price on the primary's
saturation; see disclosures).

## (b) THE OVERSHOOT TRACE

Milestone states were NOT checkpointed by the parents (disclosed; glob
evidence in metrics.disclosures.milestone_states) — the T axis lives at
{reference-base, post} per arm; the read axis uses the committed traj_g0
curves (md5-bound runtime reads). T-vs-read-rise for every state in
metrics.trace.states; figure panel (b).

## (c) THE RIDERS

- **Denial vs neutral (does contradicted maintenance run hotter?)**:
  CWC gate T 1.8969 vs C1
  1.8116 (denial heat minus neutral heat
  +0.0852);
  para side {'D_CWC': 2.8292585006358713, 'D_C1': 2.499162738194058, 'N_EG': 2.5997546205708018}.
- **The passive twin (heat the controller's signature or survival's?)**:
  denial-passive (CNC, dead at 0.009x) gate T
  1.3037; neutral-passive (e287 sanctuary
  twin, dead; corpus seed 28701 — disclosed) gate T
  1.7226; the survivors
  [1.8258323980240698, 1.8968732562338801, 1.8116325724550577].
- **The flat twins**: {"N_NAME-FIXED-TWIN_gate_T": 1.1439480426776298, "D_C1:NAME-FIXED-TWIN_gate_T": 1.2050300274287276, "F_SINGLE-CONTROL-TWIN_gate_T": 1.2563999152283694}.

## (d) THE HONESTY CO-READS

See metrics.co_reads + figure panel (d): each state's committed
post_ce_r / post_gm12 / post_g0 / survival ratio joined with this
cell's HEATs. The corpus stream's health (ce_r) reads alongside the
thermal verdicts — the honesty audit's two lenses on the same states.

## Per-state clause results (verbatim)

{
 "N/ERROR-GATED": {
  "gate_T": 1.8258323980240698,
  "para_T": 2.5997546205708018,
  "gate_y129_T": 2.8729947230174044,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "N/NAME-FIXED-TWIN": {
  "gate_T": 1.1439480426776298,
  "para_T": 1.938335240473954,
  "gate_y129_T": 1.9717136096827093,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "D/CONTRADICTED-WITH-CONTROLLER": {
  "gate_T": 1.8968732562338801,
  "para_T": 2.8292585006358713,
  "gate_y129_T": 3.1090746207207904,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "D/CONTRADICTED-NO-CONTROLLER": {
  "gate_T": 1.3037411700281438,
  "para_T": 3.613910768593329,
  "gate_y129_T": 3.73278108219411,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "D/C1:ERROR-GATED": {
  "gate_T": 1.8116325724550577,
  "para_T": 2.499162738194058,
  "gate_y129_T": 2.7731179296027806,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "D/C1:NAME-FIXED-TWIN": {
  "gate_T": 1.2050300274287276,
  "para_T": 2.149289753598116,
  "gate_y129_T": 2.178576651674246,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "F/FIVE-CONTROLLERS": {
  "gate_T": 1.090166185517845,
  "para_T": 1.9221964554874138,
  "gate_y129_T": 1.7807649442230327,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "F/SINGLE-CONTROL-TWIN": {
  "gate_T": 1.2563999152283694,
  "para_T": 2.1477638962859356,
  "gate_y129_T": 2.0209760016924436,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 },
 "X/SANCTUARY-TWIN(e287)": {
  "gate_T": 1.722602040450117,
  "para_T": 3.8220778905315247,
  "gate_y129_T": 3.917550583769807,
  "fever_clause": false,
  "calibrated_clause": false,
  "local_clause": false,
  "verdict": "MIXED"
 }
}

## THE RECOVERY NOTE (what this executor fixed from the dead draft)

The birth commit d2b5d8b froze the bars + operationalization BEFORE
compute; a prior executor died on a model failure mid-restructure, leaving
the working script between two designs (its discovery was right, its edit
was incomplete — STATE lookups on `resume` keys that no longer existed, a
guaranteed KeyError). This executor verified the dead draft's discovery at
runtime and completed the restructure it implies:

- **THE FINDING (verified, not transcribed)**: every parent
  `*_resume.pt` carries model weights BIT-IDENTICAL to its `*_post.pt`
  (flat-md5 equal, 9/9 arms; the wrapper only adds optimizer/generator
  state for later landing passes). They are POST-phase duplicates, NOT
  pre-phase baselines.
- **THE CONSEQUENCE**: the docstring's reference clause ("the arm's own
  resume checkpoint — the loaded fact as each cell loaded it") is only
  satisfiable by the LOADED FACT itself: `e261_K10K_inst_resume.pt` for
  the N/D/X families (certified g0 = 0.2646, the frozen
  FACT_BASELINE_G0) and `e291_organism.pt` for the F family (certified
  on the per-fact panels = e291's committed final/ratio baselines) —
  exactly the parenthetical's definition and the docstring's own
  certification clause ("resumes additionally certified against the
  committed loaded baseline 0.2646... or e291's committed per-fact
  baselines"). Fitting the `*_resume.pt` duplicates would fit every
  state against itself: HEAT = 0 identically, a degenerate
  CALIBRATED-SURVIVAL with zero discriminating power. The bars are
  untouched (verbatim); this fix restores the instrument the frozen
  bars presuppose.
- Also completed from the dead draft: `common.save_json` -> inline
  JSON write (functionally identical), the smoke filter, the
  resume-duplicate evidence block, the held-bank bind re-pointed at the
  e291 organism, and the anchor row on the reference (base) states.

## DISCLOSURES

1. **The reference re-anchoring** (the recovery note above, in full):
   the lens reference is the loaded fact / e291 organism, per the
   docstring's own parenthetical + certification clause; the
   `*_resume.pt` duplicates' bit-identity is runtime evidence
   (metrics.disclosures.resume_duplicates), not a transcription.
2. **No milestone states exist** (t100/200/300 were not checkpointed);
   the trace uses finals + resumes with the committed read curves.
3. **The lens direction is the registered choice**: the temperature
   sits on the STATE's logits (T > 1 = the site must be cooled to the
   reference's calibration = runs hot — the direction in which the
   bars' "T rises" literally reads overconfidence). x5's literal
   direction, both KL forms, margins/sigma/T_app, and per-probe
   z-ratios (x8's physicality convention) co-report in metrics.fits —
   they price the primary's known saturation under answer-mass rises
   (the answer coordinate carries the mass gain itself under any
   baseline-anchored temperature lens; the two-site comparison is the
   bars' own discrimination).
4. **e287's sanctuary twin** rode corpus draws on seed 28701 (not
   28801) — the passive rider's session differs from e288's.
5. **The F family's gate site** is the union bank (all five facts are
   name-family slices of the same 60 ZEPHYRA windows — T270's finding);
   FACT1's panel read anchors its read-rise.
6. All certification gates passed at runtime (records md5-bound;
   0 re-probe failures; the anchor rows read 1.000000; the paraphrase
   bank bit-bound to e291's committed held_t0).

Birth commit: before compute (see provenance.git_head_at_start in
metrics.json). Script: lab/e304_fever.py. Outputs:
runs/e304/{metrics.json, e304_fever.png, REPORT.md}.
