# INTERPRETER pass — T039-amendment (E069 D1/D2) stress-test (2026-09-26)

CPU-only reading/thinking. No THINKING/NOTES/QUEUE/STATE edits. Sources:
THINKING.md T031/T037/T038/T039/T039-amendment/T041, NOTES.md E013/E013c/
E053/E053b/E053c/E063/E069, runs/e069/metrics.json (stored curves re-read),
lab/e069_onset_discriminators.py (protocol).

---

## 0. The result under attack

Same frozen e053c ctx-512 net, same 4 generated sequences, V-zero sweep:

| read | eval-512 | eval-256 (re-indexed) |
|---|---|---|
| a* (5-pt robust, thresh 0.01 nats) | 6 [4,8] | 18 [7,30] (per-seq 8/26/30/7) |
| ages 1-3 spike (nats) | 1.74 / 4.47 / 2.32 | 2.72 / 4.89 / 2.35 |
| ages 4-5 "shoulder" | 0.91 / 0.32 | 0.06 / 0.13 |
| ages 6-17 tail | ~ -0.03 (9/12 negative) | ~ +0.02 (10/12 positive) |
| clean CE | 0.4528 | 0.4587 |

D2 (shuffled-char older half, FULL 512 window): ages-1-2 spike retention
128% [98,170]; tail STAYS DEAD, a* stays 6; clean CE 0.453 -> 0.617.

The card registered H-redistribute (attention mass re-anchors far ->
mid-recent) vs H-instrument (full-window-reference artifact). Both are
incomplete. Four facts already in the stored curves sharpen the problem
before any new mechanism is proposed.

## 1. Free facts from the stored e069 curves (no new compute)

**F1 — the shoulder COLLAPSED, it did not inflate.** Ages 4-5 fell
0.91/0.32 -> 0.06/0.13 nats. The card's "ages 4-17 go from dead to
+0.01..0.06" conflates two opposite moves: the shoulder (4-5) lost ~0.85
nats of load while the haze (6-17) gained ~+0.04. Same tokens, same ages,
same target — only the wpe code band and window composition changed.
Pure softmax renormalization from removing far keys scales all survivors'
mass PROPORTIONALLY (survivor mass ratios are invariant); it cannot make
age-1 gain +0.97 while age-4 loses -0.85. **Any uniform-mass story
(H-redistribute in its plain form) is already falsified by F1.**

**F2 — a SECOND LIVE SPIKE appeared at the WINDOW START.** The truncated
window's oldest position (age 255, wearing wpe row 0) has V-zero dCE =
+0.699 nats (CI [-0.16, +1.79]; noisy at B=4 but the 4th-largest load in
the window). The SAME token at the SAME age (255) in the eval-512 frame
(wearing wpe row 256) is dead: +0.007. The unregistered absolute-position
control completes a partial 2x2:

| far half | wpe code of that token | dCE |
|---|---|---|
| present | row 256 (native) | +0.005 (dead) |
| present | row 0 (true seq start, age 511) | -0.066 (dead) |
| absent | row 256 (abspos control) | +0.043 (dead) |
| absent | row 0 (re-indexed) | **+0.699 (LIVE)** |

Neither "row 0 is magic" nor "far-half absence" alone suffices; the live
cell is the conjunction. The 5/5 "sink dead at generation" result (T031)
holds only in the NATIVE coordinate frame — under re-indexing a dead
boundary position resurrects. The card's two registered hypotheses have
no slot for this object at all.

**F3 — the tail haze is sign-structured, not obviously noise.** Paired
per-age differences (eval256 - eval512), ages 6-17: 11/12 positive, mean
shift +0.043 nats (each age's bootstrap CI is wide at B=4, but the sign
consistency across 12 ages plus the F1/F2 reshapes argue structure). The
magnitudes are 1-4x the 0.01-nat instrument threshold — real but tiny.

**F4 — the a* = 18 is 2 sequences talking.** Per-seq a* = [8, 26, 30, 7]
vs eval-512's [8, 4, 6, 3]. The 5-pt robust rule at a 0.01-nat threshold
transduces a +0.04-nat haze into a 3x onset shift. T031 already
documented onset instrument-sensitivity (a* 3-86 across statistics;
identity broken with sign flips).

**F5 — D1 and D2 double-dissociate content vs slots.** D2 destroys far
CONTENT (keeps all 512 slots): spike amplified 128%, tail dead, a* = 6.
D1 removes far SLOTS (keeps content, re-codes positions): spike mildly
amplified (age 1 +56%), tail alive, window-start live, a* = 18.
Spike amplification is produced by BOTH far-degradations; tail/shoulder/
boundary reshaping is produced ONLY by slot removal.

**F6 — the amplification lives in the lesioned branch (arithmetic on
stored numbers).** dCE = log p_clean - log p_lesion. D2 age 2: dCE grew
+1.19 while log p_clean fell only 0.164 => log p_lesion fell 1.35 (~88%
of the move is in the lesioned branch). D1 age 1: lesioned branch fell
0.98 of the 0.97+0.006 total (~100%). The clean branch is nearly
invariant; zeroing the young spike hurts far more when the far half is
degraded.

## 2. Stress-testing exhaustiveness of the registered pair

The two registered explanations are NOT exhaustive. Candidate set:

### C1. H-wpe-domain / positional-reshape (NEW — leading)
With fewer wpe rows in play, and with late-window content wearing
mid-window codes, the circuit's per-position read policy changes
non-uniformly: the youngest read (ages 1-3) is code-band-agnostic
(relative-distance/local circuitry — induction-like), the shoulder read
(ages 4-5) is absolute-code-sensitive (native rows ~507-508 -> re-coded
251-252 -> loses its reader), and row 0 acquires a value-bearing anchor
role it lacks in native long windows.
- SUPPORTS: T038 (position code lives input-side; circuit-address with
  graded row weight); e066b (wpe rows partially load-bearing in place,
  per-donor spread — conjunction position x content); the abspos
  catastrophe (CE 6.31 > uniform 4.17 — the position code is emphatically
  NOT translation-invariant, so re-indexing is a genuine regime change,
  not a no-op); F1/F2/F5 (non-proportional reshape + boundary
  resurrection only under slot removal).
- CONTRADICTS: nothing direct; weakest point is that the primary variant's
  clean CE is fine (0.459), so the regime change must be invisible in the
  function yet visible in the per-position causal profile — exactly what
  a many-redundant-reads circuit would do, but it is an extra commitment.

### C2. H-instrument / threshold-proximity (registered #2 — partial)
The "dead tail" judgment is reference- and threshold-dependent; the a*
rule amplifies a +0.04-nat haze into a* = 18.
- SUPPORTS: T031's documented instrument-sensitivity; F4 (2 seqs drive
  the mean); magnitudes 1-4x threshold.
- CONTRADICTS: F2 (window-start +0.70 is far outside the noise band;
  no threshold rule creates that) and F3 (sign consistency). Verdict:
  explains the SIZE of the a* move, not the existence of the haze or the
  reshape. A multiplier on C1, not a rival.

### C3. H-redistribute / attention re-anchoring (registered #1 — weakened)
Fewer positions -> far attention mass re-anchors onto mid-recent ones.
- SUPPORTS: softmax physics (mass must go somewhere); the 6L atlas's
  L0 far-mass 0.80 (if similar here, L0 renormalization is substantial);
  e063's universal allocation template makes an attention-level echo
  natural (though e063 measured layer-level training-regime load, not
  eval-window geometry — the analogy is loose).
- CONTRADICTS: F1 (proportional scaling cannot drop the shoulder while
  raising age-1); the sink/far end was dead in 5/5 cells (little
  load-bearing mass out there to move — though grazing mass can still be
  large while its V-zero cost is ~0); F5 (D2 kept all slots and did NOT
  inflate the tail, so mass content is not the tail's currency).
  Survives only as a contributory term for the mild 6-17 gains.

### C4. H-competition / backup-loss (NEW — wrong for the tail, right for the spike)
dCE is a CONDITIONAL marginal cost: with the far half present, zeroing a
mid-recent V leaves the prediction backed up (far evidence floor);
truncate, and each lesion costs more with zero attention change. Clean-CE
invariance is automatic (average far value ~0, e013's 16-token
sufficiency) while conditional-on-lesion value is not (e013c bimodality).
- SUPPORTS: e013 (far context worth ~0 on average); e013c/T007 (30.6% of
  positions gain >= 0.15 from far context — the backup is real, unevenly
  distributed); F6 (amplification in the lesioned branch is its
  signature).
- CONTRADICTS (for the TAIL): F5 — D2 destroyed far content (and thus any
  backup value) yet the tail stayed dead. Backup-loss cannot be why ages
  6-17 inflated. It remains the best frame for the SPIKE amplification
  common to D1 and D2 (see section 4).

### C5. H-exposure/mismatch (minor)
Eval-256's query at wpe row 254 was trained to predict early-window
corpus tokens, never a late self-generated token. Clean CE 0.459 says the
mismatch is functionally negligible; keep only as a footnote.

Not exhaustive-proof, but C1-C5 cover every combination of {mass,
statistic, code, conditional-value} the instrument can see.

## 3. Ranking by parsimony given lab knowledge

1. **C1 H-wpe-domain/positional-reshape** — one mechanism (row-specific,
   graded, input-side position code already established by T038/e066b)
   explains F1+F2+F5 simultaneously; no new faculty required.
2. **C2 H-instrument/threshold** — explains the a* number (with T031
   precedent) but not F2/F3; rides on top of C1.
3. **C4 H-competition/backup-loss** — correct for the spike (both D1 and
   D2), dead for the tail (F5).
4. **C3 H-redistribute** — plain form falsified by F1; survives as a
   minor contributory term only.
5. **C5 H-exposure** — footnote.

## 4. The single cheapest discriminator (protocol, registerable)

**One measurement: per-position attention-received mass of the final
query under eval-256 vs eval-512, same net, same 4 sequences, plus K-drop
vs V-zero at three probe ages.** One battery of CPU forward passes,
minutes, zero training, uses runs/checkpoints/e053c_ctx512.pt verbatim.

- WHAT: for each window (512 native; 256 re-indexed), at the final
  position, record attention-received mass summed over heads, per layer
  (4x4 table), for ages 1-20 and the window-start token (age 255 in the
  256-frame). Also run the K-drop variant of the sweep at ages 4, 5, and
  window-start.
- REGISTERED SIGNS:
  - C1 H-wpe-domain predicts a RESHAPED profile: attention(ages 4-5)
    falls under eval-256 (>= 25% relative), attention(window-start/row 0)
    rises >= 2x, ages 1-3 within +/-20%; window-start K-drop >> V-zero
    if it is anchor/sink mass, ~equal if it is a value read.
  - C3 H-redistribute predicts proportionally SCALED survivors: all
    surviving ages gain mass by roughly the same factor
    (~1/(1-far-mass)), including ages 4-5.
  - C2 H-instrument predicts no attention change anywhere (|d| < ~5%);
    if that holds while the dCE reshape stands, the tail story returns to
    the statistic and the window-start spike must then be explained as a
    V-side fluke of 1-2 sequences (testable in the same run by per-seq
    breakdown).
- REGISTERED NUMBERS (falsifiable): attention(window-start, eval-256) /
  attention(same token, eval-512) >= 2.0 AND attention(age-4, eval-256) /
  attention(age-4, eval-512) <= 0.75 => C1. All ratios in [0.9, 1.1] =>
  C2 (statistic) for the tail + spike-amplification frames (C4).
  Uniform ratios > 1.1 => C3.
- SECONDARY (same run, free): an in-distribution 256 control — the
  window = positions 0..255 of the same sequences (native codes 0..255,
  target = position 256's token, which was generated with exactly that
  history). If a*(native-256 slice) is ~6 with a dead window start, the
  D1 effect is specific to truncating a LONG window (code-band /
  composition), not to window length per se — killing the last refuge of
  a pure window-fraction story.

This supersedes the card's registered discriminator (which only contrasted
"mass grows" vs "no change" at ages 4-17) by adding the two sign-bearing
probes (ages 4-5, window-start) that F1/F2 show are the informative ones.

## 5. D2 interpretation — leaning vs interference-removal

The card's frame: "with distant context scrambled the net leans harder on
recent positions" (attention re-allocation). The proposed alternative:
interference-removal (far entries net-negative; removing them improves
recent reads).

**Interference-removal is falsified by the card's own manipulation
check.** Scrambling the far half RAISED clean CE 0.453 -> 0.617: at these
targets the far context was net-POSITIVE evidence (+0.16 nats), and this
net's measured old-entry interference is tiny (-0.01..-0.05 nats/position,
vs the +1.76 nats of extra ages-1-2 load). You cannot buy +1.76 nats of
spike amplification by removing -0.03-nat interference. e013c/T007's
loser-tail (28.2% of positions lose >= 0.15 from far context) is real but
is a different net/instrument, and the winner side dominated here; T007's
registered shuffled-far prediction ("both tails collapse if
structure-driven") is in fact partially executed by D2 — and the observed
outcome (clean CE up, tail dead, spike up) is what a tug-of-war won by
the evidence side looks like.

**"Leaning" is unverified and, in its plain form, unnecessary.** F6 shows
the amplification is 80-100% in the LESIONED branch with the clean branch
nearly invariant — exactly what CONDITIONAL-REDUNDANCY (floor-loss)
predicts with NO attention re-allocation: the far half is a backup floor
of ~0 average value (e013) but large conditional value (e013c's 30.6%
gainers); destroy it and the same young-spike lesion crashes the
prediction harder. dCE is a conditional statistic, and the far half's
removal raises the young positions' CONDITIONAL marginal value while
leaving their unconditional (clean) contribution intact. The same frame,
not "leaning", also explains D1's age-1 inflation (+0.97 nats) — two
different far-degradations, one common conditional-marginal-value effect
(F5's spike row).

**Verdict: switch the frame — not to interference-removal but to
conditional-redundancy / floor-loss** (backup-evidence removal). Keep
"leaning" only as the explicitly-registered fallback: it survives iff the
section-4 measurement shows attention(ages 1-2) growing under scrambled
far context. The registered dissociation: floor-loss predicts
attention(ages 1-2) ~unchanged (scrambled keys still soak grazing mass)
while log p(t | young-V zeroed) does all the falling; leaning predicts
the mass itself migrates.

## 6. Paper bearing (one paragraph)

The window-invariance claim should be restated as a three-way split:
(a) the young spike HORIZON (~6 tokens) is invariant across eval windows
and n-gram statistics (D2's H1 verdict stands); (b) the per-position
causal PROFILE beyond the spike is coordinate-frame-sensitive — shoulder,
tail, and boundary loads all move when the wpe band changes, with clean
CE invariant (function robust, causal atlas not); (c) a* as a NUMBER is
statistic-sensitive near its threshold (T031's caveat, now with a second
instance). e053c's truncation claim carries the eval-window qualifier, as
the card already says — but the qualifier's mechanism is now testable
with one registered CPU measurement, and the "dead sink" result needs the
phrase "in its native coordinate frame".
