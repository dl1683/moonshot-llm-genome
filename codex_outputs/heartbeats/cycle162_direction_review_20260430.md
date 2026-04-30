**Ranked Recommendations**

1. **Run g194 next.** A17 is the live SEV-10 blocker. g192 can only say “the row-identity effect survives at 28 layers,” but if g194 shows the effect is mostly norms/frequency scalars, then “content” was overclaimed. g194 should factorialize `e_t = r_t * u_t`: correct norm + correct direction, correct norm + random/shuffled direction, shuffled/uniform norm + correct direction, scalar-only controls, same Fro, same mask, same anchor.

2. **Keep §0.1 at 4.5/10.** g193 does not lower the score because g191 still stands: exact matched row identity under anchor is real and shuffled rows are harmful. But g193 kills the +2.4 uplift. No compiler law, no move to 5.5-6.0.

3. **Best interpretation of g193 FAIL:** yes, it is meaningful, but narrow. The finding is: trained embedding rows contain useful exact-string interface content, but cheap token-form statistics cannot synthesize their directions. This strengthens the “interface codebook + decoder” story and weakens any “learnable lexical geometry law” story. Be precise: g193 falsifies byte histogram + length + log-frequency, not every possible contextual/distributional compiler.

4. **Pivot direction after g194.** The string-match thread is now near diminishing returns. The big-lab-unlikely angle is not “we copied shared vocab rows”; it is a negative architecture law: representational charts are architecture/interface-specific, wrong rows poison training, and exact lexical identity is not compressible into simple token features. Turn that into an **interface compatibility diagnostic**: predict which tokenizer/init/anchor choices will help or harm before paying training cost.

5. **g194 vs g192:** g194 moves §0.1 more.  
   **g194: 8/10 leverage.** If directions carry the signal, §0.1 can rise toward ~5.0 because “content” becomes real directional content. If scalars carry it, the project avoids a false narrative.  
   **g192: 6/10 leverage.** Useful only after g194 direction-pass. It improves scale robustness, but does not resolve the central ambiguity.

Bottom line: **g194 first, then conditional g192. Do not run g190 next.** g190 is mechanism exploration after validity is secured, not the next score-moving experiment.

