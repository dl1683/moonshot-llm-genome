# THE SIGN-FRONT NULL — the pure desk derivation (2026-10-02, zero compute)

The R61-critic's forced item (scratch/r61_critic.md, attack 1 + the forced
experiment), dispatched as T166's provisional stamp demands. Zero GPU, zero
training; the only compute is the scalar arithmetic shown (desk checks,
reproduced in-session). Every input number is committed and gated: e194, e196
(via e197/e201), e197, e200, e201 metrics.json + e_chart's census (NOTES
~L519/L578) + THINKING T164/T166. Desk: theory agent, 2026-10-02.

INPUT LEDGER (all committed, all gated):
- Alive-pair lag-1 cosines (e201 census, adjudicated): org1 (u0,u1) −0.15453;
  MIRABEL (u0,u1) −0.17511, (u1,u2) −0.17814; org2-half (u0,u1) −0.26318,
  (u1,u2) −0.31285, (u2,u3) −0.34129, (u3,u4) −0.35280. NOTE: the census holds
  SEVEN alive pairs (the task brief said six and listed seven); all seven used.
- Lag-2 cosines (committed, never before read as a class): half u0·u2
  +0.14055, u1·u3 +0.23746, u2·u4 +0.14246 (e200 ray_geometry); MIRABEL u0·u2
  −0.01570, u1·u3 +0.07932 (e201); org1 u0·u2 +0.03151 (e194 front_trace
  state-2 vs_static_sign = cos(u2,u0)).
- Lag-3/4: half u0·u3 −0.03702, u1·u4 +0.01719, u0·u4 −0.02586 (e200); org1
  u0·u3 +0.00290 (e194 state-3 vs_static_sign); MIRABEL u0·u3 −0.00252 (e201).
- Post-death context: org1 (u1,u2) −0.20298 → (u2,u3) −0.18565 → (u3,u4)
  −0.15233 → (u4,u5) −0.14851 (e194 continuation); MIRABEL (u2,u3) −0.22036
  (e201 u3); org2-dead-full (u0,u1) −0.16531, (u1,u2) −0.07544 (e196 via
  e197/e201) — all flagged, never adjudicated.
- Steps (T139: AdamW sign step L2 = lr·√P): org1/MIRABEL 1.6542880 over
  P = 2,739,072 → per-coordinate s = 9.996e-4; org2 0.9164196 over P = 873,472
  → s_full = 9.806e-4, s_half = 4.903e-4 (same lr convention to 2%).
- Gradient scale at t=0: org1 preclip gnorm 0.98292 (clip 1.0 not binding) →
  per-coord RMS 5.939e-4; org2 preclip 2.29925 (clip BINDS) → post-clip L2
  1.0 → RMS 1.070e-3. Sign fronts are clip-invariant (positive rescaling).
- Walk gnorms (preclip): org1 0.98 → 6.19/3.93/4.49/4.26/4.54; MIRABEL 0.95 →
  6.93/6.34(/5.52 at u3); org2-half 2.30 → 5.88/3.39/4.80/8.22 (e194/e201/e200
  journals).
- Chart census (e_chart/T150, NOTES ~L578): at the root, top-0.1% of |g|
  coordinates carry 40% of ||g|| (linear); top-10% carry ~75% of ||g||²
  (opt2 energy_frac 0.7497; e_chart's own convention reads 86.6%); bottom-90%
  typical |g| ≈ sqrt(0.25/0.9) ≈ 0.53 × RMS (opt2 convention; 0.39 × RMS at
  e_chart's). Exact-zero gradient coordinates at t=0: org1 960 (0.035%);
  org2 33,408 (3.8%) — e194 G_R2DIR / e197 G_SIGNRAY.

---

## 1. THE CLOSED FORM (the critic's sketch confirmed, then corrected)

Assumptions, exact:
- A1 (local quadraticity): over one step, the bath-mean gradient μ(θ) is
  per-coordinate linear: μ_{t+1,i} = μ_{t,i} − h_i·s·sign(g_{t,i}), with h_i =
  ∂μ_i/∂θ_i ≥ 0 the per-coordinate curvature (diag Hessian; cross-terms and
  their error fold into the bath term). This holds only for steps that stay
  inside the local basin — see §4 for where it is violated on record.
- A2 (bath decomposition): the read gradient g_t = μ_t + ε_t, ε the per-batch
  draw, per-coordinate, independent of μ, with lag-k sign-agreement
  a_k = E[sign ε_t · sign ε_{t+k}] (a_0 = 1; for disjoint corpus windows
  a_k ≥ 0 small, and a_1 ≥ a_2 ≥ a_3 … — no mechanism makes adjacent-batch
  draws anti-correlated).
- A3 (sign step): θ_{t+1} = θ_t − s·sign(g_t), s the per-coordinate size
  (A3 holds by measurement: T139's lr·√P arithmetic, gated).
- A4: exact-zero coordinates are a negligible class (0.035% org1; 3.8% org2 —
  the latter flagged as texture).

DERIVATION. Per coordinate, sign(g_{t+1,i}) = −sign(g_{t,i}) iff the step
crosses the per-coordinate minimum: |g_{t,i}| ≤ h_i·s (overshoot); else same
sign. Partition coordinates into:
  F (flip, signal-dominated): |μ| ≤ h·s and the flipped residual |μ|−h·s
      survives the bath → contributes −1 to the lag-1 sign product;
  Sh (shell, signal-dominated): h·s < |μ| ≤ 2h·s — same sign at lag 1, flipped
      at lag 2 (crosses one step later) → +1 at lag 1, −1 at lag 2, +1 at lag
      3 (period-2 thereafter, deterministic);
  L (large, signal-dominated): |μ| > 2h·s → +1 at all short lags;
  N (noise-dominated): the post-step residual is below the bath sd → the sign
      is redrawn: contributes a_1 ≈ small at lag 1, a_2, a_3 … at longer lags.
Then, with f = f_F, f_sh = f_Sh, f_n = f_N, and 1−f−f_sh−f_n−a-terms = f_L:

  cos₁ ≡ cos(u_t, u_{t+1}) = 1 − 2f − f_n + f_n·a_1  (exactly: 1 − 2f − f_n(1−a_1))
  cos₂ ≡ cos(u_t, u_{t+2}) = 1 − 2f_sh − f_n(1−a_2) + (shell's lag-2 flip already in; L,F agree: +f_L+f −f_sh−f_n(1−a_2))
       = 1 − 2f_sh − f_n(1−a_2)
  cos₃ ≡ cos(u_t, u_{t+3}) = 1 − 2f − 2f_sh·0 … = f_L − f + f_sh + f_n·a_3
  cos₄ → f_L + f − f_sh + f_n·a_4 (period-2 core: even lags positive, odd negative)

THE CRITIC'S FORMULA IS CONFIRMED at lag 1: cos(u_t,u_{t+1}) = 1 − 2·f_flip −
f_n, with f_n the near-zeroed/noise-decided fraction (the "newly-zeroed"
class is the boundary shell of the flip zone: coordinates crossing their
per-coordinate minimum land near |μ| ≈ 0 and join N). The a_1 correction is
small and POSITIVE (bath correlation can only push cos₁ toward 0/+): a
negative cos₁ cannot be produced by the bath in any spec — deliverable (iv)
holds; the negativity is the algorithm's own step, localized in F.

THE PERIOD-2 FINGERPRINT (new, desk-only, decisive). The deterministic core
makes the lag sequence alternate sign (odd lags negative, even positive,
decaying as noise erodes); pure bath correlation gives a_k ≥ 0 decaying
smoothly with NO sign alternation. The committed matrices read (alive pairs):

  half:  lag1 −0.263/−0.313/−0.341/−0.353  lag2 +0.141/+0.237/+0.142
         lag3 −0.037/+0.017   lag4 −0.026
  org1:  lag1 −0.155            lag2 +0.032            lag3 +0.003
  MIRABEL: lag1 −0.175/−0.178   lag2 −0.016/+0.079     lag3 −0.003

Negative odd lags, positive even lags, ≈0 by lag 3–4: the period-2 signature,
in the committed data, unread until now. T164's "a rotating object" is
mis-named: a rotation with a fixed period would shift phase smoothly across
lags; a coordinate-wise sign oscillation at period 2 is what the data show.
The alternation IS the algorithm's bounce — now fingerprinted, not assumed.

## 2. THE (f_flip, f_n) PLANE — the critic's parameter point is refuted; the corrected point

The censused lag-1 values are reproduced along lines f_n = 1 − 2f_flip − cos₁.
The critic's sketch (f_flip ≈ 0.58–0.68, f_n ≈ 0) sits on those lines — at
f_n = 0: 0.577/0.588/0.589/0.632/0.656/0.671/0.676 for the seven alive pairs.
But the plane is degenerate in lag-1 alone; the committed lag-2/3 break it:

REFUTATION OF f_n ≈ 0 (any f_n ≲ 0.3). With f_n = 0, the mass budget forces
f_sh = (1−cos₂)/2 and f = (1−cos₁)/2 − f_sh, giving a DETERMINISTIC lag-3
cos₃ = f_L − f + f_sh ≥ +0.2 in every solvable system:
  org1 t0: f=0.093, f_sh=0.484, f_L=0.423 → predicts cos₃ = +0.814; observed +0.003.
  MIRABEL t0: f=0.080, f_sh=0.508, f_L=0.412 → predicts +0.841; observed −0.003.
  half t0: f=0.202, f_sh=0.430, f_L=0.368 → predicts +0.596; observed −0.037.
The f_n ≈ 0 specification is dead by 15–280× at lag 3. The gradient-mass
plausibility the critic appealed to was never the problem; the lag structure is.

THE DATA-SELECTED REGION. Solving the (cos₁, cos₂) systems with a_1 ≥ a_2
(normal for adjacent draws) makes the naive estimator a LOWER bound on the
deterministic flip class: f ≥ (cos₂ − cos₁)/2:
  org1 t0 ≥ 0.093;  MIRABEL t0 ≥ 0.080, t1 ≥ 0.129;
  half t0 ≥ 0.202, t1 ≥ 0.275, t2 ≥ 0.242.
and f_n = 1 − cos₂ − 2f_sh + f_n a_2 ∈ [~0.66, ~0.95] across feasible specs
(f_L ≈ 0 forced within the c=0 family; closing the residuals needs a small
positive adjacent-bath agreement a_1 ≈ 0.05–0.15 — consistent, since lag-3/4 ≈ 0
bounds a_3, a_4 ≤ 0.04 and nothing anti-correlates adjacent draws).
STATEMENT: the censused anti-correlations are carried by a DETERMINISTIC
MINORITY (the overshoot flip class, ~8–28% of coordinates, organism- and
step-dependent) riding on a NOISE-REDRAWN MAJORITY (~66–95%). The critic's
headline arithmetic ("they ARE the floor") survives in corrected form: banal
parameters reproduce every censused value AND the full lag structure the
critic's point could not.

Plausibility vs the chart census: bottom-90% coordinates carry ≤ 25% of ||g||²
(top decile 75–86.6%), i.e. the typical coordinate sits at |g| ≈ 0.4–0.53 ×
RMS — a large small-magnitude majority exactly of the shape f_n requires (the
bath redraws them), while the flip class (~0.1–0.3) plausibly lives in the
small-|μ| body plus moderately curved coordinates (|μ| ≤ h·s with s = 1.68 ×
RMS_org1, 0.46 × RMS_org2-postclip). The census does NOT pin h (curvature is
unmeasured everywhere in this record) — the f values are consistent with the
mass distribution, not derived from it; that honesty stands.

## 3. THE ABSORPTION RETRODICTION (the −0.263 → −0.353 sequence)

Deterministically, a coordinate above threshold falls |μ| → |μ| − h·s per step
and, once inside |μ| ≤ h·s, never leaves (period-2 lock-in). So the flip set
at pair t is the CDF of |μ₀|/h evaluated at (t+1)·s: f_flip(t) = Pr[|μ₀,i| ≤
(t+1)·h_i·s] — monotone in t, with increments given by the swept tail. The
model therefore RETRODICTS: (i) monotone deepening of |cos₁| along a walk
(absorption), (ii) shrinking increments (a heavy tail thins as it is swept),
(iii) the same behavior at dead states (the bounce is state-blind).

Naive reading (f_n = 0): f(t) = 0.632/0.656/0.671/0.676; increments
+0.0248/+0.0142/+0.0058 — monotone, concave, exactly the committed sequence's
shape; "absorption rate" ≈ 1.5% of coordinates per step, decaying.
Corrected reading (lag-2 pairs available t = 0,1,2): the deterministic core
0.202 → 0.275 → 0.242 — NET growth (+0.04 over two steps ≈ +8%/step net) but
NON-monotone at t=2, tracking the walk's gradient-mass swings (gnorm
5.88/3.39/4.80/8.22 — a 2.4× batch-to-batch wobble that rescales the |μ|
distribution and hence f under fixed s). Verdict: the retrodiction is
QUALITATIVELY consistent (net absorption across the alive window, mandatory
post-death persistence, shallowing of org1's post-death pairs −0.203 → −0.149
as the wrecked state's gradient mass explodes and dilutes the core) but not
quantitatively exact without letting the mass wobble — which the committed
gnorms say it does. T164 read this sequence as "a rotating object, not a
converging pursuit"; the null reads it as the bounce core slowly fattening
under a wobbling gradient mass. No fact, support, or front required.

## 4. THE STEP-SIZE PREDICTION — no committed in-domain test; the one two-size pair reverses direction

The model's only sharp, parameter-free qualitative prediction: the flip zone
|μ| ≤ h·s widens with s, so the deterministic core (and |cos₁| at matched
states and steps) must be NON-DECREASING in s within one organism, in-domain.

The single committed pair readable at two sizes: org2 (u0,u1), same root
(e157_f2), same licensed stream (seed 10902, same batch draws, same u0 to the
md5): full step (s = 9.806e-4, step L2 0.9164) → −0.16531; half step
(s = 4.903e-4, step L2 0.4582) → −0.26318. DIRECTION REVERSED: the LARGER
step reads SHALLOWER, opposite the model's in-domain prediction. The larger
arm is out of domain: org2's full step (0.9164) is 1.74× its own u0 static
sign-ray kill edge (0.5252; e193/e197 committed) — θ₁ sits past the kill cliff
(primary ruler 0.0068, dead), where A1 is void and the wrecked state's
gradient mass shifts the whole (|μ|, σ) distribution. The half arm is in
domain (0.458 < 0.525). So: NOT a falsification, but NOT a confirmation — the
step-size law has zero committed in-domain evidence, and the null must not be
stamped as if it did. Cross-organism magnitudes (half lineage deeper than
org1/MIRABEL at 2× the per-coordinate step) cannot arbitrate: the nets differ
(org2's postclip per-coord RMS 1.070e-3 vs org1's 5.939e-4 — the s/RMS ratios
are 0.46 vs 1.68), and h is unmeasured. This debt is collected in the
falsifier cell's ladder (below).

## 5. THE REGISTERED FALSIFIER (for the queue; draft registration)

DRAFT CELL — e202 SIGNFRONT-NULL (CPU-only, threads 4, four ≤5-step sign walks
+ battery reads ≈ the e201 u3-burst class, ~3–4 min, chunkable ≤180 s; owner-
envelope legal; e194/e201 conventions verbatim: u_t = sign(g_t)/||sign(g_t)||,
the licensed seed-10902 stream, tiered identity gates, bars frozen before
compute).

ARMS. (A) org2 ladder: sign walks at s* ∈ {1/2, 1/4, 1/8} × 0.9164 =
{0.4582, 0.2291, 0.1146} — every rung strictly inside the 0.5252 static sign
edge (in-domain), 5 steps each. (B) FACT-FREE TWIN: e048_repro.pt (org1's
pre-install 2.74M base — same architecture class, no fact), full-step 1.6543
walk, same stream, 5 steps. READS: all fronts u0..u4, the full mutual matrix
at lags 1–3, per-step preclip gnorm, per-step battery; alive-window gating per
arm's own ruler convention (the fact-free arm reports all steps, flagged).

BARS (draft, frozen before compute):
- PERIOD-2-UNIVERSAL (the null's shape): every arm, every alive t: cos₁ < 0
  AND cos₂ > 0 AND |cos₃| ≤ 0.05 (gnorm band < 3× t=0 disclosed per step).
- LADDER-MONOTONE (the model's step-size law, in-domain for the first time):
  the anti-phase core (cos₂ − cos₁)/2 at t=0 non-decreasing in s across
  {1/8, 1/4, 1/2}×. FAIL = the overshoot model's in-domain prediction is
  falsified and the null needs re-derivation (this bar can kill the null
  itself — registered as such).
- FACT-IN-THE-FRONT (the rescuer): fires iff the fact-free twin's alive-pair
  cos₁ at matched (architecture-class, s, t) deviates from the fact-carrying
  twins' committed values (org1: −0.1545; the s/2 rung vs half −0.2632) by
  ≥ 0.05 with the SAME SIGN at ≥ 2 consecutive pair indices, WHILE the
  fact-free arm itself passes PERIOD-2-UNIVERSAL (so the deviation is the
  fact's, not a broken null). DIRECTION OF RECORD: the fact-carrying walks
  DEEPER (cos₁ more negative / core larger) = the fact contributes
  deterministic anti-phase mass. FACT-FREE-ON-CURVE: all matched |Δ| < 0.05 →
  the alternation carries no fact information and T166's headline deflates.

ONE-LINE FORM: FACT-IN-THE-FRONT fires iff a fact-free sign walk (pre-install
net, same stream, matched step) shows alive-pair lag-1 cosines ≥ 0.05 away
from the fact-carrying twins in the direction that makes the fact's fronts
deeper — otherwise sign descent bounces, as it must.

## 6. THE VERDICT (ruthless, both ways)

DEFLATED — fully, as evidence of anything about the fact:
- "ALTERNATION-UNIVERSAL" (e201's verdict) and T164's "the front is a
  ROTATING OBJECT": the committed lag structure (−/+/≈0 at lags 1/2/3–4) is
  the period-2 fingerprint of sign-descent overshoot; the −0.10 bar is passed
  by construction by any sign walk at these s/g scales with a ≥ 5% flip zone.
- "THE ROTATION OUTLIVES THE ORGANISM" (T166's headline): mandatory, not a
  discovery — the bounce is state-blind; org1's post-death pairs actually
  SHALLOW (−0.203 → −0.149) with the wreck's gradient-mass growth, another
  algorithmic texture misread as biology.
- The deepening sequence as "a rotating object, not a converging pursuit":
  retrodicted as absorption (+ mass wobble) with no fact required.
- The "256× the isotropic floor" framing: the wrong null, as the critic said;
  the relevant floor is the overshoot curve, and the values sit ON it.

SURVIVES (untouched by this null — different instruments, already bracketed
by the critic's other attacks):
- DEATH-AT-DEEPEST-LANDING (T164's residual): the 4-point rank-order
  (ratios 3.17/0.663/1.292/0.397 vs ruler deltas +0.43/+0.12/−0.32/−0.48,
  perfect) is a D_kill object; the overshoot null says nothing about the
  lethality of directions. n = 1 lineage, event-censored, support-proxy
  overlap at the kill −0.030 — a within-lineage regularity awaiting a second
  lineage, not a noun.
- The onset/arrival asymmetries (T163): D_kill-ratio objects; bracketed by
  attack 2 (bar 0.60 vs 0.70; 4.3× denominators), not by this null.
- The k-ladder re-computation bonus (e194: refreshed fronts kill at 1.75 vs
  frozen 2.27): lethality-of-adaptivity, orthogonal to the alternation.

THE NULL'S OWN DEBTS (honesty reflex on the null itself): (i) its only sharp
in-domain prediction (step-size monotonicity) is UNTESTED on committed data —
the one two-size pair reverses direction from OUTSIDE the quadratic domain
(full step = 1.74× its own kill edge); (ii) h (curvature) and the bath
agreement a_k are unmeasured — the corrected parameters are consistency
regions, not measurements; (iii) therefore the null is LIVE AND
FINGERPRINT-CONSISTENT, not yet stamped: run e202's ladder + fact-free twin
(minutes, CPU) before spending another rotation-adjacent noun. T166's
provisional stamp should now read: "consecutive sign fronts anti-correlate at
the magnitude sign-descent overshoot predicts, with the period-2 lag
fingerprint, on every lineage measured; the open question is whether the
killing step's root-anchored lethality peak carries fact information — and
whether ANY of the alternation's parameters move when the fact is removed."
