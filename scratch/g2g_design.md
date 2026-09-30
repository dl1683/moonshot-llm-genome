# g2g — THE RHYTHM'S CONTROLS (design draft v1, 2026-09-30 ~11:18Z)

Status: DESIGN-DRAFT; bars frozen here for the dispatch docstring.
Builds on: R56 critic attack C3 (construction disclosures), T131's
amendment (the debts), T136 (timing 3/3 roots; amplitude = ruler-geo
alignment), W023 (the replay-as-mini-shock mechanism candidate), the
critic's fixed-arm co-read (0.693 vs 0.587-0.615). What is NEW: every
prior g2 run held wash intensity constant — the "self-timed" claim
has never met a threat ladder, a lowered refractory, or a registered
fixed-period head-to-head.

## The three questions

1. THERMOSTAT OR CLOCK: does the organ's event rate respond to the
   wash's threat level, or is it a threshold+cooldown oscillator
   indifferent to threat? CONFOUND NAMED UP FRONT: the threat dial IS
   the wash lr — threat and clock speed are one variable at this
   instrument; the discriminator is the ORGAN's rate response (does
   the interval track the wash's decay speed?), not the fact's.
2. WAS THE BAND REAL: with REFRACTORY lowered from 24 to 8 (below
   the band floor 20), spacings <20 become POSSIBLE — the in-band
   fraction becomes a measurement instead of a construction.
3. DOES SELF-TIMED EARN ITS KEEP: the registered head-to-head vs a
   fixed replay schedule at matched event count — the critic's
   co-read (fixed 0.693 vs organ 0.587-0.615) becomes an adjudicated
   bar, not a footnote.

## The cell (locked root, organ verbatim, wash seed 10902 family)

ARMS: (a) threat ladder wash-lr in {0.5x, 1x, 2x, 4x} x the organ
verbatim — dense readout per rung (the g2c convention), event
intervals + monitor decay slopes; (b) refractory {8, 24} at 1x (the
band-widening control); (c) the fixed-period arm at matched event
count (1/32 family) vs the organ at 1x — both cycle-medians on the
same ruler; RIDER (W023, cheap): the monitor trace around replay
events — post-event decay-slope change (the mini-shock signature:
sharper dip, slower recovery) at every event, both refractory
settings.

## Registered bars (frozen)

  SELF-TIMED-THERMOSTAT: "fires if event rate is monotone in wash-lr
      with >=2 distinct rates across the ladder — the organ responds
      to threat; the rhythm is not fixed-period."
  FIXED-PERIOD-ARTIFACT: "fires if event rate is constant across the
      ladder — the rhythm is a threshold+cooldown oscillator at one
      threat level; the 'self-timed' wording is retired to
      'event-driven at constant threat'."
  SELF-TIMED-WINS: "fires if the organ's cycle-median exceeds the
      matched-count fixed schedule's by >= 0.05 at >=2 threat
      levels."
  FIXED-MATCHES-OR-WINS: "fires if the fixed schedule is within
      noise or better — the organ's claim narrows to autonomy (zero
      scheduling signal, zero parameters), performance equal;
      reported honestly in T131's amendment and the paper."
  REFRACTORY-REAL: "fires if, at refractory 8, >=1 event spacing
      lands < 20 — the band claim was refractory-bound (disclosed as
      a construction artifact); if all spacings still >= 20, the band
      reflects the wash's own decay clock and stands as measured."

## Honesty pre-registration

n=1 root (the locked root; T136's redraw rungs stay separate), one
wash seed per rung first cell; the threat ladder conflates threat
with clock speed (named above — the reading is the organ's RESPONSE
curve, never "threat caused"); the fixed arm uses the critic's
co-read lineage (matched event count, not matched replay batches —
batch parity co-reported if the fixed arm's replay differs in batch
composition); replicate ladder (seeds) only if SELF-TIMED-THERMOSTAT
fires.

## Envelope

GPU for the wash rungs (4 ladder arms + 2 refractory arms + fixed
arm, each <= 180s + cooldowns; ~20-30 min total) — queues behind
g1bS per C13-2 (no further g-series cell before g1bS; g2g IS the
critic-forced control, queued after); dense CPU evals per the g2c
convention.
