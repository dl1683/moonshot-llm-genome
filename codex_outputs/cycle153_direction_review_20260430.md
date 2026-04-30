Yes. The direct string-match result changes the §0.1 trajectory, but it does **not** mean “go back to g188 OT.” It means the live signal is probably **exact lexical row identity**, not flow geometry.

Recommendation: finish the current g188 controls, then pivot to string-match decomposition before running vanilla g190. g190 remains important, but only after it is redesigned to control for matched vs unmatched rows. Otherwise a g190 PASS could just be rediscovering the 84% exact-token overlap.

Top 3 by §0.1 uplift:

1. **String-match decomposition: 8/10 uplift potential**
   Run `direct_init_only`, `direct_anchor_only`, matched-rows-only, unmatched-rows-only, row-shuffled matched map, frequency-bucket shuffle. If +0.49 holds across seeds and lives in anchor-only/matched rows, §0.1 probably moves from 3.2 to ~4.6-5.0. It gives a clean mechanism, though not yet a breakthrough.

2. **g190, but stratified: 7/10**
   Run decoder-conditioned relearning only after adding direct-string baseline and unmatched-row reporting. Biggest win: g190 improves **unmatched rows** or beats direct-string at equal row budget. That would support “decoder-compiled codebook,” not just shared spelling.

3. **Prospective tokenizer-compatibility law: 6.5/10**
   Use exact-match %, frequency-weighted overlap, row norm/spectrum compatibility, and matched/unmatched anchor gain to predict which tokenizer swaps help before training. This is the path from artifact to diagnostic product.

Bottom line: **decompose string-match first; do not abandon g190, but do not run it blind.** The direct match result is the first positive cross-tokenizer bridge signal, and it should now become the control surface for every next experiment.

