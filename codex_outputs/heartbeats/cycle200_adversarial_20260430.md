1. **SEV-10: active output-head regularizer, not transferable content.** g191 already says anchor-only gives ~98% of the gain; g195 says the dominant surface is `lm_head`. The clean kill story is: a continuous tether to a pretrained classifier basis improves CE while present. It may leave no learned residue.

2. **SEV-9: “direction” may be scaffold geometry, not token content.** g194 kills norms, but not the possibility that any well-conditioned angular basis aligned to Qwen coordinates works. Random/shuffled directions are toxic controls, not sufficient positive controls.

3. **SEV-9: exact-string lookup table, not a law.** g193 failing the compiler strengthens this objection: the effect may be an uncompressible table over shared token strings, not transcodable interface geometry.

4. **SEV-8: C4/token-frequency coupling.** The output head directly touches next-token CE. If gains concentrate on high-frequency exact-match rows and do not survive OOD eval, the “training health” framing is too broad.

5. **SEV-7: g192 is not evidence yet.** One matched seed at 28 layers is encouraging, but without all seeds and full-depth shuffled/scaffold controls, it cannot close the shallow/regime objection.

**Resolving experiment:** run a compressed **28-layer untied output-only residue/scaffold kill**: `scratch`, `output_anchor_full`, `cutoff_500`, `cutoff_2000`, `late_anchor`, `orthogonal_scaffold_full`, `cov_scaffold_full`, plus C4 and OOD eval. Falsify the strong claim if cutoff arms fall below +0.12 nats or scaffolds recover >=50% of active-anchor gain. Pass only if `cutoff_2000` retains >=45% of gain, 3/3 positive, scaffolds dead, and OOD positive.

