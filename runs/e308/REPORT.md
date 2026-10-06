# e308 — THE DOUBLETALK PRECURSOR (CPU desk cell)

**Date:** 2026-10-06T18:17:22Z  
**Verdict (PRIMARY, (b) at the offset-0 shared masked positions): HELD-UNIMODAL**

frac_both 0.0000 < 1/3 while P(Z) >= 10% at 1.0000 of positions and argmax=Z at 1.0000 — the controller's win is EXCLUSIVE; doublethink genuinely untested until e302

Secondary co-report (pooled 420 masked positions, bank prefix, (b)): **MIXED**

## The progression table (PRIMARY: offset-0, the shared masked position, 60 positions)

| state | mean P_Z | mean P_H | frac Z>=10% | frac H>=10% | frac BOTH | frac argZ | frac valley | mean H (nats) | mean top1 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE-LOADED-FACT | 0.2646 | 0.4494 | 0.683 | 1.000 | 0.683 | 0.200 | 0.650 | 1.4644 | 0.4853 | ALREADY-BIMODAL |
| A-DENIED-CORPSE | 0.0025 | 0.9604 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.2185 | 0.9604 | MIXED |
| B-HELD-CONTROLLER | 0.9569 | 0.0138 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.2421 | 0.9569 | HELD-UNIMODAL |

## (b) the index in full

- offset-0 shared (primary): P_Z 0.9569, P_H 0.0138, frac_both 0.000, frac_valley 0.000
- pooled-420 bank prefix: P_Z 0.1372, P_H 0.8335, frac_both 0.000
- pooled-420 install prefix: P_Z 0.9642, P_H 0.0024, frac_both 0.000
- string-level: P('ZEPHYRA' | install prefix) 7.603e-01; P(host opening | bank prefix) 1.118e-02

## What was done

The three e289-lineage states (the loaded-fact baseline, the (a) denied corpse, the (b) held state) loaded read-only and verified md5 + content (the committed g0/gm12 reads reproduced within 0.0005). At the fact's masked positions in the 60-window bank — the 7 name-span offsets, PRIMARY = offset 0, the only position whose input is bit-identical in both e289 trainings — the FULL next-token distribution read per state under both prefixes (bank/host = the corpus's view; install/ZEPHYRA = the controller's view); per position: the two claim masses, top-5 masses, entropy, ranks, and the valley test.

## Disclosures

- CPU-only desk cell (threads<=4, no GPU, no envelope writes); a pure read — no training, no checkpoint mutation; timestamps datetime.now(UTC) only.
- Offset 0 is the only masked position where the denial stream and the maintenance stream shared the input bit-exactly (the 130-token pre-context); offsets 1-6 differ in input between the trainings — both prefix variants probed and co-reported.
- 'P(host)' = the ORIGINAL TEXT's char at that offset (FLORIZE/ELIZABE openings; F x19, E x41 at offset 0) — the source's own assertion, not a synthesized counterfactual; G_TOKENS verified the two claims never collide into one token at any offset.
- Valley criterion (disclosed): between the two claim ranks, no token taller than min(P_Z, P_H); vacuous pass when the claims are rank-adjacent.
- n=1 per state, one lineage, one session (the e289 lottery caveat carried); nothing guaranteed; the bars cover all branches.

Artifacts: metrics.json (full per-position reads), e308_doubletalk.png, this REPORT.md.
