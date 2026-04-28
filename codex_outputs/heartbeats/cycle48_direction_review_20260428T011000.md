Q1 decision tree:
A: run g173 immediately.

B: insert `g175_component_anchor_attribution` before g173. Do a coarse, energy-matched trained-vs-random/permuted dissection: embeddings/head, early/mid/late blocks, attention, MLP, norms. Score by null-subtracted gain, not raw gain. Do layers first; low-rank/magnitude-band tests are second pass after localization.

C: retire C18 as donor-transfer. Do not pivot the main project to generic L2 regularization unless a standard weight-decay / L2-to-init control shows a real efficiency edge. Main line becomes KD/late-KD cash-out; weight-anchor becomes an optimizer artifact note.

Q2: yes, elevate g172 as its own claim line regardless of g174. Wording: “late-stage KD recovers 69% of full-KD gain with 33% KD-active compute; early KD retains only 14%.” Standalone §0.1 weight: 6.8/10. It becomes ~7.1 if g174 Part B passes donor-specificity; ~6.2 if random-teacher KD matches trained.

Q3: add a short caveat/status banner now, full rewrite after g174. The synthesis is already stale because it predates g167/g170/g172 and donor-specificity controls. Do not rewrite from scratch until A/B/C is known.

Q4: no recorded procurement update found. Wall-power remains blocked; if acquired, g155 jumps to top.

Queue:
- g174: 9.5/10. Finish; it gates the narrative.
- g173: 8.2/10 only on A or KD-specificity survival; defer on B until component attribution.
- g155: 8.2/10 scientific ceiling, 0/10 executable until meter exists.

One concrete change: add `g175_component_anchor_attribution` as the explicit B-branch queue item, so a close contest does not default to g173 prematurely.