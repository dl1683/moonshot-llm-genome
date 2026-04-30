## Cycle 155 Adversarial Review (A16) -- g188 direct_string_match claim

**Codex session:** 019dddd2-b6ca-7461-93d0-1bc1569f0564
**Reviewed:** g188 direct_string_match_anchor +0.478 nats mean (3/3 seeds) = 93.2% of g181b

### Ranked attacks

1. **SEV-10: This is not cross-tokenizer transfer; it is shared-vocabulary pretrained-row reuse.** With 84.1% exact token-string overlap, the arm mostly says "GPT-2-tokenizer Qwen shell trains better when most rows are initialized/anchored to trained Qwen rows for identical strings." That is useful, but the word "transfer" is overclaiming. The 16% unmatched rows are mean-filled, then the whole matrix is Frobenius-rescaled, so the effect is not isolated to matched-row content yet.

2. **SEV-9: Decoder-conditioned geometry is not established.** The decoder shell remains Qwen3-shaped. A positive result can be explained by trained Qwen embedding rows being good lexical priors under a Qwen3 decoder, not by any tokenizer-transcoding law. OT and char-overlap failing strengthens the simpler story: exact row identity matters; geometry transport does not.

3. **SEV-8: 8-layer/5000-step scale may be a shallow-init regime.** The result may vanish or shrink in the full 28-layer model where deeper dynamics dominate. g181b proves persistence for the same-tokenizer 8-layer setting, not full-model cross-tokenizer persistence.

4. **SEV-8: Shuffled harm is not enough.** flow_shuffled_qwen_rows = -0.724 shows wrong rows are toxic, but that only proves token-row identity matters. It does not distinguish "semantic trained content" from frequency/norm/token-prior lookup-table effects.

5. **SEV-7: Alternative ruled out too fast:** exact string overlap may be a data-distribution shortcut. If the frequent C4 mass is concentrated in the shared 84%, the result could be almost entirely frequent-token loss improvement, with unmatched rows irrelevant or even hidden by evaluation weighting.

### Resolving experiment

Run g191, but make the decisive criterion stricter: matched_rows_only must recover most of +0.478, unmatched_rows_only must be near zero, and both row-shuffle plus frequency-bucket-shuffle must stay near zero. Then repeat only the winning matched-row arm on the full 28-layer shell. If full direct works but matched-only fails, or frequency-bucket shuffle survives, kill the "trained content transfer" claim and relabel it "shared-token lexical initialization prior."

### Impact on g191

- g191 already tests attacks #1 (matched vs unmatched decomposition) and #4 (shuffled controls)
- Attack #5 (frequency-bucket shuffle) is g191 arm 7
- Attack #3 (28-layer test) requires a follow-up experiment AFTER g191 -- not currently planned, should be pre-staged as g192 if g191 PASS_CONTENT
- Attack #2 (decoder-conditioned) is g190's territory -- deferred until g191 clarifies mechanism
