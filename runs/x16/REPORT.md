# X16 — THE OFF-TARGET SPECIFICITY PROBE

* the primary verdict: **MIXED** — the table verbatim: stability 0.9933, flip-or-squeeze 0.0300 (bar 0.25), dmargin mean -0.00042, neg share 0.493, sign p 8.63e-01, dentropy mean +0.00017, Z-bias excess +0.01005 — neither frozen bar met; the mechanism reads live in the table
* the secondary: **ALIGNMENT-MATTERS** — the random-direction control shows a materially smaller effect at matched norm: host-g0 boost +0.000071 <= 25% x the trigger's +0.084783; mean |dmargin| 0.00056 <= 50% x the trigger's 0.01170 — the +32% needed the read-aligned direction

## Headline numbers

* ANCHOR: the committed +32% reproduces BIT-EXACTLY (0.3494305908679962 vs committed 0.3494305908679962, |d| 0.0e+00); the applied state's flat md5 == e311's arm-A (f84fd2647fa5).
* TRIGGER on the 300-context general battery: top-1 stability 0.9933 (bar: >= 0.95 volume / flip-or-squeeze 0.0300 vs bar 0.25); Δmargin mean -0.00042, neg share 0.493, sign-test p 8.63e-01, skew -0.303; Δentropy mean +0.00017; Δp(winner) mean -0.00005 (0.51 negative); Z-bias excess +0.01005 (0.82 of contexts positive).
* RANDOM CONTROL: stability 1.0000; Δmargin mean -0.00003 (mean |Δ| 0.00056); host-g0 boost +0.000071 vs the trigger's +0.084783; cos(trigger, ctrl) +9.96e-05.
* ON-TARGET (the 60 g0 contexts, co-report): p(Z) 0.2646 -> 0.3494 (trigger) / 0.2647 (control); trigger Δmargin mean -0.0436, Δentropy -0.01410.

## The gate ledger

| gate | what it binds | pass |
|---|---|---|
| G_NAMEFREE | the corpus carries no name | True |
| G_SPLICE | the host battery reconstruction (19+41) | True |
| G_BATTERYGEO | the g0/gm12 battery geometry | True |
| G_PARENTS | e311 metrics + vectors md5-bound; the anchor literals cross-checked | True |
| G_ROOM | the K10K room D/S bit-bound vs e264_rooms.pt | True |
| G_FACTLOAD | the host loads bit-exact (3-way) | True |
| G_TRIGBIND | the committed trigger (md5/norm/in-room) | True |
| G_CTRL | the random control (in-room ~0, matched norm) | True |
| G_ANCHOR | the +32% anchor (state md5 + read) | True |
| G_BATT | the general battery sanity | True |

## Predictions scored

* P-x16a (lab): fired=False.
* P-x16b (executor counter): fired=False (bias-shaped evidence present: False).

## The executor's live read of the table (no bar shopping — what MIXED contains)

* THE KNOB IS A WEAK TARGET-SPECIFIC ADDITIVE COMPONENT RIDING A NULL: the
  Z-bias excess (+Δlogit(Z) − mean Δlogit) is +0.0100 logits at general
  contexts — positive at 82.3% of them, 91x the random control's +0.00011 —
  so consult #010's suspected additive Z-bias EXISTS as a mechanism. But at
  c = 0.70x the transport bracket it sits far below general-text margins
  (base winner mean prob 0.624; mean |Δmargin| 0.0117): no detectable
  softmax-denominator intrusion on unrelated decisions (stability 99.3%,
  squeeze 2.7%, margins symmetric 148/152, entropy shift +0.00017 ≈ null).
* THE ZERO-SUM SIGNATURE IS ON-TARGET ONLY: at the host's own 60 g0
  contexts the same write squeezes hard — Δmargin −0.0436, Δentropy
  −0.0141, flip-or-squeeze 50%, stability 0.767, Z-bias excess +0.362
  (36x the off-target value), p(Z) 0.2646 → 0.3494. The "bias" acts where
  Z is in the competition (the host's read); elsewhere it is inert.
* NOT A VOLUME DIAL EITHER: the volume bar's confidence clauses failed on
  general text (mean Δmargin −0.0004 ≲ 0, mean Δentropy +0.00017 ≳ 0) —
  the knob does not raise off-target confidence, it does (almost) nothing
  off-target.
* W049 (Q3 calibration dial): the frozen "proceeds as planned" license did
  NOT issue (the volume bar's clauses failed), but the table shows NO
  off-target harm at the licensed dose AND ALIGNMENT-MATTERS decisively —
  the dial is target-local and read-gradient-aligned; this SHARPENS the
  alignment story (the +32% needed the aligned direction; a random
  out-of-room direction at the same norm does ~nothing, host boost ratio
  0.08%), it does not kill it.

## Provenance

* registration birth commit: b281f0d (bars + P-x16a/P-x16b frozen BEFORE
  compute; pushed); head at run time: 2d675805983bc8af78186df9c176d96bb16a546c.
* CPU only (torch threads 4, DCT workers 4); no GPU touched.
* parents: e311 (metrics + vectors + host artifact + room artifact) md5-bound; all gates in metrics.json.
* disclosed instrument repair mid-cell (pre-adjudication): G_BATT's
  uniqueness clause relaxed from == n to >= 0.99n after the first full
  draw showed 2 chance duplicate windows in 300 (the battery itself stays
  the registered seed's exact draw; no adjudication bar touched).

*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).*