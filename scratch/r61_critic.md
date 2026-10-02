# R61 CRITIC — the flight/rotation story attacked (2026-10-02, read-only, zero compute)

Mandate: attack T164/T166 (the rotation account), T163 (onset arrival), T165
(g1bS2-4, the scale saga), the owner-envelope compliance question, and name
the one forced experiment. Builds on: e194's iso-floor convention (the only
null on record), T139's Adam sign-step arithmetic (step L2 = lr·sqrt(P),
committed at 2.74M and 10M), T155's in-span spread and ruler-biography
findings, and the committed metrics of e196-e201 / g1bS2-S4 (all re-read for
this review; every number below is from runs/*/metrics.json or the frozen
scripts, not from prose).

---

## ATTACK 1 — THE ROTATION ACCOUNT (T164/T166): the null is named wrongly, and under the right null the alternation is a theorem of the algorithm, not a property of the wash

**What is on record.** Consecutive fronts u_t = sign(g_t)/||sign(g_t)||,
where g_t is the stream's step-(t+1) batch gradient read at theta_t, and
theta_{t+1} = theta_t − s·sign(g_t) with s = one AdamW step (measured
1.6543 raw at 2.74M = lr·sqrt(P) to 3 decimals; 0.9164 at 873k; 0.4582 for
the half-step lineage — the walk IS a per-coordinate sign step). Censused
pair cosines: org1 (u0,u1) −0.1545; MIRABEL −0.1751/−0.1781; half-step
−0.263/−0.313/−0.341/−0.353. Bar: −0.10 (ALTERNATION-UNIVERSAL). The only
null anywhere in the record is the **isotropic floor 1/sqrt(P)** (e194's
script line 82; NOTES line 337: "256x the isotropic floor").

**Why that null is the wrong one.** The iid floor answers "could two random
sign vectors be this anti-aligned?" — obviously not, at 250σ. Nobody's
hypothesis says the fronts are iid. The fronts are consecutive batch
gradients **straddling the walk's own sign step**. Bath correlation (same
corpus, adjacent batches) can only push the pair cosine POSITIVE — it can
never produce −0.15 "trivially". The trivial producer is the step itself:

*The desk null (sign-step overshoot).* On a locally quadratic landscape
(diag curvature h_i), g_{t+1,i} = g_{t,i} − h_i·s·sign(g_{t,i}), so

  sign(g_{t+1,i}) = sign(g_{t,i}) · ( +1 if |g_{t,i}| > h_i·s else −1 ),

and cos(u_t, u_{t+1}) = 1 − 2·f_flip + noise terms, where f_flip is the
fraction of coordinates with |g_i| ≤ h_i·s — i.e., coordinates inside their
per-coordinate bounce zone. A converged net washed at lr = its own
pretraining peak has most coordinates gradient-small; f_flip ≈ 0.58–0.68
reproduces every censused value (−0.155, −0.175, −0.26..−0.35) with banal
parameters. This is the classical period-2 orbit of sign-normalized
descent. **The observed values are not 256x above the relevant floor; they
ARE the floor.**

**The fingerprint is already in the committed data, misread.** The
overshoot model predicts monotone absorption: each step pushes more
coordinates into the bounce zone, so consecutive-pair anti-correlation
should DEEPEN along a walk. The half-step lineage's committed sequence is
−0.263 → −0.313 → −0.341 → −0.353 — monotone deepening, exactly the
absorption signature. T164 reads this as "a rotating object, not a
converging pursuit"; the overshoot account reads it as progressive 2-cycle
lock-in, with no fact, support, or front required. The "−0.10 bar" was
never derived from any null; under the overshoot null ANY sign walk on ANY
net on ANY stream passes it — including a plain fine-tune on clean corpus.
The census therefore demonstrates only that these are sign walks.

**"The rotation outlives the organism" is the null's mandatory behavior.**
A period-2 bounce continues at dead states by construction; the post-death
persistence (org1 −0.203+, MIRABEL u3 −0.2204) is presented as the day's
deepest read and is instead what the algorithm does regardless of the
fact. T166 concedes "the rotation is the wash's, not the fact's" and then
mints a headline from the concession.

**What actually survives, and how thin it is.** The residual science is
T164's *death-at-deepest-landing*: the killing step's front had the lowest
root-anchored ratio (half lineage: 3.17 / 0.663 / 1.292 / **0.397**-kill;
and the four ratios rank the four per-step ruler deltas perfectly:
+0.43/+0.12/−0.32/−0.48 — a real within-lineage dose-response, credit
where due). But: (a) n = 1 lineage with 4 points for the ranking; org1 and
MIRABEL contribute one informative front each (org1's "deepest" is its
only non-definitional front — 1 bit; MIRABEL's t1 is right-censored, never
killed inside the D ≤ 3.0 grid); (b) the sequence STOPS at death — "deepest
landing" is measured over a sample censored at the event, so a
deeper-landing-with-survival can never be observed by construction;
(c) the mechanism noun is unmeasured at the killing step: the killing
front's overlap with the only support proxy ever computed (root fact
gradient) is **−0.030** (u4; u3 −0.077) — the front does NOT land on any
measured support direction; "the fleeing support" is inferred from the
lethality it is supposed to explain. The story's two halves run on
incompatible instruments: alternation (pair cosines — overshoot-confounded)
and concentration (D_kill ratios — n=1 ranking), glued by an unmeasured
object.

**Required repair (words, before any noun survives):** re-state T166 as
"consecutive sign fronts anti-correlate at the magnitude sign-descent
overshoot predicts, on every lineage measured; the open question is whether
the killing step's root-anchored lethality peak carries fact information" —
and register the overshoot null's falsifier (see the forced experiment).

---

## ATTACK 2 — THE ONSET STORY (T163): arrival times are bar-contingent and normalized against denominators that differ 4.3x with a 2–3x direction lottery

**The comparison.** "org1 arrives t=1 (ratio 0.171), MIRABEL t=2 (0.427),
half t=2 (0.663) — the same curve shifted one step." Ratio = D_kill(u_t
ray from ROOT) / D_kill(u0 ray from ROOT); arrival = first t below the
concentration bar.

**(a) The bar moved inside the same arc, and the half lineage's arrival
sits exactly on the seam.** FLIGHT_RATIO_BAR = 0.60 in e196; 0.70 in
e197/e198/e199/e200. Half's t2 ratio is 0.663: concentrated under 0.70,
NOT concentrated under 0.60 — under e196's own bar the half lineage's
first arrival is t4, the killing step, which collapses "arrive-then-die"
into "arrive-at-kill" and dissolves the "one-step shift" picture. The
arrival ordering "org1=1, MIRABEL=2, half=2" is not a property of the
organisms; it is a property of where the bar was set in the last three
registrations.

**(b) The denominators are not comparable.** The u0 edges are 2.2699
(org1), 1.9471 (MIRABEL), 0.5252 (half) — a 4.3x spread. T155 committed
that in-span direction draws alone spread D_kill by 2.0–3.0x, and e194
showed grid coarseness moves an edge ~10% (2.5 → 2.27). A GLOBAL ratio bar
applied to self-normalized quantities whose denominators carry a 2–3x
lottery is uncalibrated: "0.427" and "0.663" may be the same depth in each
organism's own currency. The lab already owns the correct instrument and
did not use it here: each organism's own in-span random-ray band
(e189/e190's B2 convention, e192's random band). Arrival should be defined
against THAT per-organism reference (or a within-organism z-score over the
in-span spread), not a universal 0.7.

**(c) The clock is defensible only by accident.** Steps differ 3.6x in raw
size (1.654 vs 0.458). Because step/edge happens to be near-constant
(0.73/0.85/0.87 edge-units per step), counting steps is accidentally close
to counting edge-normalized movement — but that coincidence is unremarked,
and if it broke (a fourth organism), the step-count comparison would
silently break with it. State the axis: arrival in units of each
organism's own step/edge, or don't compare.

**(d) Report curves, not first-crossings.** Three organisms with 1–4
informative points, one censored, one straddling the bar: the comparable
object is each ratio trajectory, not the first-crossing integer. T163's
"ONSET-COMMON" is a first-crossing claim and inherits (a)–(c).

**Minimum repair:** one line in T163/T166 stating the sensitivity —
"under e196's 0.60 bar the half lineage arrives at its killing step" — and
a re-computation of arrival against per-organism in-span bands when those
bands exist for MIRABEL/half (they exist for org1 from e189/e190).

---

## ATTACK 3 — THE SCALE SAGA'S ENDPOINT (T165): a "landscape" built from one realization, two stability regimes, and no error bar at any dose

**The three points.** 0.0010 (0.30 rms @ lr 1e-3, g1bS2), 0.6498 (0.12 rms
@ 4e-4, g1bS3), 0.2523 (0.30 rms @ 4e-4, g1bS4); implicitly 0.042 at
~zero movement (the shared post-install read).

**(a) The 1e-3 point is not on the movement axis.** g1bS2's consolidation
was in a blown stability regime (CE stuck 2.58; the channel killed INSIDE
consolidation); g1bS3/S4 settled (CE 1.70/1.64). g1bS5's own docstring
keeps it "SEPARATE as the stability casualty" — but T165's headline
("formation landscape NON-MONOTONIC in movement") stitches it back in.
With the diverged point excluded, the stable-regime evidence is monotone
DECLINE in movement (0.6498 → 0.2523); the rise side of the non-monotonicity
rests on the implicit zero-movement point (a pre-consolidation state, not a
dose).

**(b) The two same-lr points are one trajectory, not two doses.** g1bS4
replays g1bS3's steps 1..300 verbatim (same seed 10901) and extends to 750:
0.6498 and 0.2523 are s=300 and s=750 of a SINGLE realization. And g1bS5's
three interior doses are nested prefixes of the same stream (its own
registration: "steps 1..375 of every run REPLAY g1bS3/g1bS4's committed
trajectories"). The entire 10M formation curve — all five abscissas — is
one jitter draw. Formation variance at 10M has NEVER been measured; at
2.74M the same-recipe amplitude lottery was ~1.2x (g2e/g2f) and g1bS4's
same-recipe replay fuzz is mean 0.022 / max 0.111 — the max fuzz is 43% of
the 0.6498 reading and **2.2x the SHARP-OPTIMUM margin (0.05)**. As
registered, g1bS5 can fire SHARP-OPTIMUM on fuzz.

**(c) The ruler lottery rides.** T155: a fact can be alive on its own
ruler and dead on an imported one; in-span draws spread 2–3x. The g-12
channel the whole curve is read on is a displaced-geometry battery whose
own draw sensitivity at 10M is unknown.

**Is the reading licensed?** As a TRAJECTORY statement — "on this one
seed, the channel forms by s300 and decays by s750" — yes, and it is a
genuinely interesting texture (consolidation over-writes). As a LANDSCAPE
statement — "non-monotonic in movement, sharp optimum between 0.12 and
0.30" — no: no independent doses, no replicate, no error bar, two regimes
mixed.

**What single-variant curve would license it:** g1bS5 is exactly the right
instrument (movement-only sweep at fixed licensed lr 4e-4, root reads
only, no arms — the cheapest form) — but as registered it needs ONE
addition to be decisive: a single redraw at one interior dose (e.g. 0.20
rms with a fresh jitter seed, ~500 steps of 4e-4, chunkable ≤180s under
the envelope) to put the first error bar on 10M formation. Without it,
SHARP-OPTIMUM vs MONOTONE-DECLINE is unfalsifiable at the committed fuzz
floor. Also pre-commit now that the 1e-3 point stays off the curve.

---

## ATTACK 4 — THE OWNER-ENVELOPE ERA: compliance is asserted by the agent that has an incentive to assert it

g1bS4's metrics record the envelope as POLICY STRINGS ("util<=20% AND
temp<=70C double-polled before every burst; >=180 s cooldown") and NOTES
adds prose ("181s cooldowns; 5 heat pauses ridden"). No measured poll
values, no timestamps. e201 records one number (cpu_load 36% at launch).
Every future claim — "we stayed under 60-70% total util", "never
back-to-back", "cooldowns honored" — currently rides on self-report.

**The one-step fix (code-written, tamper-evident by git, zero science
compute):** make `common.py:gpu_status()/gpu_ok()` APPEND every poll to
`runs/_envelope_log.jsonl` — {ts, experiment_id, util, mem, temp, power,
decision(ok/wait/park)}. Then:
- burst boundaries and durations are the gaps between records of the same
  caller (the envelope's 60–90s cap and ≥120–180s cooldowns become
  auditable arithmetic);
- "double-polled" is two adjacent records <5s apart;
- the reviewer/auditor diffes JSONL vs the NOTES claim; committed with
  metrics, so history polices edits.
This is a function body inside the existing guard — not a new automation
(the heartbeat-cron rule is respected). Interim, no-code check available
TODAY: progressive phase timestamps inside metrics.json (e.g. g1bS5's
`phases` + `date` fields) already segment bursts; an auditor can partially
reconstruct duty cycle from committed artifacts alone. Recommend the JSONL
before the next GPU cell (g1bS5's cooldowns are running RIGHT NOW and are
currently unverifiable).

---

## THE FORCED EXPERIMENT (weighed under the shared, fragile machine): the sign-front null, as a pure desk derivation — zero compute

Everything empirical is expensive under the owner envelope; this one is
free and re-bases the bars of a whole arc:

**Derive E[cos(u_t, u_{t+1})] for sign-normalized walks.** Quadratic local
model + bath noise: cos = 1 − 2·f_b − f_n (f_b = deterministic flip
fraction |g_i| ≤ h_i·s; f_n = noise-dominated fraction). Deliverables:
(i) the closed form and the plane of (f_b, f_n) that reproduces the six
censused values; (ii) the ABSORPTION LAW — pair anti-correlation deepens
monotonically along a walk as coordinates enter the bounce zone — and its
check against the committed −0.263→−0.353 sequence (already consistent;
state it as a retrodiction); (iii) the step-size prediction — halving s
shrinks f_b, so the half-step lineage at LATER t (more absorbed) can still
be more negative than org1 at its first pair, dissolving any
cross-organism reading of the magnitudes; (iv) the statement that bath
correlation alone cannot go negative (kills the "trivially −0.15 from
bath" worry in the direction that matters, and localizes the negative cos
in the algorithm's own step).

**Registered falsifier for T166's residual claim (no compute):** under
overshoot, the alternation magnitude and its absorption are functions of
(steps taken, step size, gradient/curvature scale) ONLY — so the committed
pairs from ANY sign walk on these nets (e.g. a neutral-stream wash, a
no-fact net) must lie on the same curve. If a future cell finds pair
cosines that deviate from the overshoot prediction *in the direction of
the fact's presence* (e.g. alternation measurably weaker/stronger on the
extinction stream vs a fact-free stream at matched states), the rotation
carries fact information; otherwise T166's headline deflates to "sign
descent bounces, as it must."

Cost: literally zero GPU/CPU seconds — pencil, the committed numbers, and
the e194/e201 conventions. Under the owner envelope it is the only
experiment this review should force.

---

## Repairs requested (in priority order)

1. T166: strike or re-derive "THE ROTATION OUTLIVES THE ORGANISM" against
   the overshoot null (Attack 1); add the censored-sample caveat to
   "death = the deepest landing"; state that the killing front's overlap
   with the only measured support proxy is −0.03.
2. T163: add the bar-sensitivity line (0.60 vs 0.70 moves half's arrival
   to its killing step); commit to per-organism in-span normalization
   before any further "arrival" language.
3. T165/g1bS5: pre-commit the 1e-3 point stays off the movement axis;
   append one redrawn interior dose when the envelope allows; note
   max-fuzz 0.111 vs the 0.05 SHARP-OPTIMUM margin in the registration.
4. BEFORE the next GPU cell: the `_envelope_log.jsonl` append in
   `common.py` (Attack 4).
5. Dispatch the desk derivation (the forced experiment) as a thinking
   card; gate any further rotation-noun on its outcome.
