# e075 design — source-aware pruning: the T045 intervention test (P3 CACHE WEATHER, step 1)

Status: DESIGN (Rule-0 memo). CPU-only (e053c net, generation is cheap).
Register before run; frozen bars below.

## Why

T045 (e073+e074, closed): negative-utility cache entries concentrate in
the model's OWN generated tokens (4/4 nets; late-gen 5.5× early-gen;
prompt entries ~never hurt even when shuffled). The 2025 KV-pruning
literature evicts by attention/recency/sink — never by token SOURCE.
The natural intervention test: prune by source and measure.

## Arms (fixed-anchor protocol, B=8, e053c net, seeds documented)

1. **A-none (control):** normal free run 64→512.
2. **A-self-prune:** from generation step ~100 on, every K=32 steps,
   V-zero the model's own generated entries with age > 96 (the dead
   band beyond the live window + shoulder).
3. **A-prompt-prune (placebo):** V-zero CORPUS-prompt entries at the
   same matched ages/times (63 old entries — the band T045 says is
   harmless).
4. **A-both-prune:** both bands (report-only; total-cache-size
   control).

## Registered readouts + bars (frozen)

- R1 tail clean CE (final 64 tokens): A-self-prune ≤ A-none + 0.01
  nats (no cost, possibly gain) AND A-prompt-prune ≥ A-none + 0.05
  (placebo hurts). BOTH firing = source-specific pruning value.
- R2 fluency: entropy/top-k drift ≤ 5% in A-self-prune.
- R3 the per-age utility curve AFTER pruning (does pruning self-old
  shift the live spike? predict: unchanged).
- Kill: A-self-prune costs > +0.05 nats → the "dead weight" is not
  prunable at the value level after all — report as the honest
  boundary of the T045 implication.

## Envelope

CPU minutes-scale (B=8 × 448 tokens × 4 arms, batched). No training.
One file lab/e075_source_prune.py; runs/e075/ outputs.
