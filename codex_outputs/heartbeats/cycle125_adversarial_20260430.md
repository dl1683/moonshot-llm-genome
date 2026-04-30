1. **S10: the remaining signal may be alpha/protocol response, not geometry.** g182’s universal forecast claim is dead; the survivor is only `KD - scratch` delta at one KD dose. If KD alpha=1.0 happens to create a consistent architecture-specific NLL shift, geometry deltas may just decode “how hard did this protocol perturb optimization?” The strongest falsifier is g186: geometry must beat `alpha-only`, `alpha^2`, `delta_early_loss`, `delta_telemetry`, Shesha, and `(arch, alpha)` arm means on held-out seeds and held-out doses. If alpha/telemetry ties, kill the claim.

2. **S9: C23 strengthens the confound, not the forecast story.** The latest canonical PASS says the durable effect is Qwen3 tokenizer/embed/lm_head prior held in place, not internal structure. That makes the pairwise delta more likely an interface-compatibility/KD-mismatch detector than a general “training health geometry” law.

3. **S8: n=24, 8-ish geometry features, post-failure salvage.** R2=0.518 is interesting, but it was discovered after catastrophic LOAO failure, permutation failure, and arm-demeaned failure. Treat as hypothesis-generating unless g186 pre-locks the exact feature set and survives seed-block inference.

4. **S8: hidden dose leakage.** Geometry deltas may nearly decode KD strength. That is only meaningful if geometry predicts residual outcome after dose is known. Add an explicit alpha-decodability diagnostic; if geometry predicts alpha well and alpha predicts outcome equally well, the geometry claim collapses.

5. **S7: Shesha/ordinary telemetry may erase novelty.** If public geometry-stability metrics or mundane early optimization telemetry match the delta result, the project has no distinctive moat, only a local reproduction of “representations encode training dynamics.”

**One resolving experiment:** run g186 as a strict residual dose-response kill: 2 architectures x 6 seeds x alpha `{0, .3, .7, 1, 2}`; primary target `final_NLL(alpha=0)-final_NLL(alpha=d)`; primary model frozen to preregistered geometry deltas; PASS only if geometry beats the best non-geometry baseline by a material margin on held-out seeds and held-out alpha levels, with seed-block CI > 0. Fail that, retire Forecast as central.

