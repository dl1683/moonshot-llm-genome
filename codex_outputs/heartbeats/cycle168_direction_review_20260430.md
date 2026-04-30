**Q1. g194 Interim**

Direction is winning, but not fully proven yet. `correct_dir_shuffled_norm` sitting between scratch and full_match means correct token directions carry substantial signal even when per-token norms are wrong. That weakens the scalar-norm explanation.

Caveat: until `shuffled_dir_correct_norm` and `random_dir_correct_norm` finish, this is direction-favorable, not PASS_DIRECTION. Also, tied `lm_head` means this may still be output-classifier geometry, not purely input embedding geometry.

**Q2. g195 Or g192 First**

Run **g195 first**.

Reason: g192 inherits the tied-head confound. A 28-layer PASS would only say “the tied input/output row prior persists at depth.” It would not tell you whether the mechanism is input embedding, output logits, or coherence between both.

g195 has higher information value because it can kill or rescue the mechanism framing directly. g192 is important, but it is robustness after mechanism clarity.

**Q3. g196 Anchor-Cutoff**

Yes, but do not make it only “remove anchor at step N.” Make it an **anchor-residue factorial**:

`init_only`, `anchor_only`, `init+anchor`, `anchor_cutoff_50/500/2000`, and ideally `late_anchor_only`, using the winning g194/g195 arms.

Decision rule: if correct directions help only while tethered and vanish after cutoff, this is regularization. If they leave persistent gain after cutoff, you have basin/content residue. Add random orthogonal / conditioned angular-basis controls, because A18 correctly warns that “direction” may mean a well-conditioned scaffold.

**Q4. If g194 + g195 + g192 All Pass**

Honest §0.1 score: **~6.0/10**, maybe **6.2** if margins are clean.

That would be a real result: coherent input-output trained token-row directions at exact lexical matches improve training and persist at full Qwen3 depth. But it is still capped because it remains exact-string, within-family, anchor-dependent, and not yet a training-health predictor.

Highest-leverage next experiment: **g196 anchor-residue** if not already done. If g196 also passes, then move to a prospective **interface compatibility diagnostic**: predict which tokenizer/init/anchor interventions help or poison training before running them, against early-loss baselines and low-overlap tokenizers.

**Q5. Honest Assessment**

Right now, this is mostly **within-family tokenizer/interface prior**, not yet “earliest geometry predicts training health.”

But it is not useless phenomenology. It is carving out a causal variable: interface row geometry can help or poison optimization. To become the live §0.1 thesis, it must predict held-out training outcomes better than early loss or simple protocol labels. Until then, call it **anchor-conditioned interface compatibility**, not training-health forecasting.

