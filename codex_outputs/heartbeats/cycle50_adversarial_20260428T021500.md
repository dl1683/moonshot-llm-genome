Strongest attack: **7/10**.

The claim is overstated, but not dead.

1. **Permuted +0.128 nats is probably above zero, but tiny.** Per-seed gains are `+0.179, +0.123, +0.081`, mean `+0.128`, SD `0.040`. A paired 3-seed t interval is roughly `[+0.029, +0.227]`, so it is not just noise on seed variance. But “12% of trained” is a descriptive ratio, not a proven mechanistic decomposition. With `n=3`, do not read it as “12% generic regularization.”

2. **Random harm does not prove donor-specificity.** This is the best objection. Random anchor harm only proves random weights are a bad attractor. It does not separate “specific Qwen donor information” from “any trained/pretrained-like weight basin.” The current result supports **trained-structure specificity vs random/permuted nulls**, not donor identity specificity.

3. **Embedding/head permutation concern mostly does not kill it.** Code applies hidden-axis permutations to tensors with hidden size, and metadata confirms `model.embed_tokens.weight` was permuted on axis 1. I found no separate `lm_head.weight` in saved applied metadata, likely because it is tied/omitted from `named_parameters`. Still, the vocab rows were not permuted, so token-specific embedding geometry is partly preserved. That could plausibly explain some or all of the `+0.128` residue.

4. **500 steps is a real scope limit.** The donor arm is still far from the donor NLL. This proves an early-training / finite-budget advantage, not convergence persistence. “Persistent” here means persistent through 500 SGD steps, not asymptotic.

5. **Generalization is too broad.** The evidence is one architecture, one donor, one dataset, one matched-null lambda. g165 shows trained-donor effect at `λ=0.0013`, but g174 did not test permuted/random nulls there. So donor-specificity is locked only for `Qwen3-0.6B`, C4, `λ=0.01`, 500 steps.

**Single falsifying experiment:** run the same 3-seed protocol with a **different trained, same-shape Qwen3-0.6B-family anchor**, matched for Frobenius distance and gradient norm, plus true donor, scratch, permuted, and random. If the non-donor trained anchor gets `>=80%` of the true-donor gain, or true donor fails to beat it by a practical CI-separated margin, the “donor-specificity” claim dies. The active ingredient would be “trained-like weight basin,” not donor-specific information.