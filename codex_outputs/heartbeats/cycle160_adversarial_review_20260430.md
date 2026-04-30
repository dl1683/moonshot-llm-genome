1. **SEV-10: scalar-vs-direction confound remains open.** g191 kills *global* format: spectrum/Fro/table distribution. It does **not** kill per-token scalar format. `row_shuffled_matched` misassigns row norm, frequency-correlated magnitude, string/length priors, and direction all at once. So the result supports “correct row identity matters,” not yet “directional trained content matters.”

2. **SEV-9: shuffled harm may be toxic interference.** A wrong anchored row is worse than no prior because SGD is continuously forced toward the wrong lexical embedding. The -0.709 gap is compatible with “misaligned anchor poisons training,” not positive evidence that the aligned row contains transferable content.

3. **SEV-8: g191 mostly proves anchor dominance.** `direct_anchor_only` ≈ `matched_rows_only` ≈ full signal. That means the mechanism is the continuous tether, not warm-start transfer. The claim should be narrowed to: “anchoring exact-match token rows to Qwen3 trained rows is beneficial in this 8L shell.”

4. **SEV-8: frequency_bucket_shuffle is weak evidence.** It preserves only coarse frequency quintiles. Within-bucket shuffling still destroys exact frequency rank, row norm alignment, token length/morphology correlations, and lexical identity. With only 1/3 seeds done, it cannot carry much interpretive weight.

5. **SEV-7: lexical lookup-table explanation survives.** Exact-match Qwen3 rows may simply be a pretrained lexical prior under a Qwen3 decoder family. That is useful, but not a broader trained-content transfer law.

6. **SEV-6: 8-layer/5000-step scope limit.** g191 can pass locally and still fail as a full 28-layer model result. g192 is required before claiming architecture-relevant persistence.

**Resolving experiment:** run a scalar/direction factorial on matched rows: decompose each row as `e_t = r_t * u_t`. Arms: correct `r_t` + correct `u_t`; correct `r_t` + random/shuffled `u`; uniform or shuffled `r` + correct `u`; scalar-only random directions. Same row mask, same anchor, same Fro. PASS_CONTENT only if correct directions recover most of +0.465 while correct scalars with wrong/random directions stay near scratch. Right now, g191 has not isolated that. 
