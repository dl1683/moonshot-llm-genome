I’ll do a quick memory pass for prior `moonshot-llm-genome` context, then read the named repo files directly so the critique is grounded in the current artifacts.

Memory says the old strongest claim had already narrowed to a trained-text activation-spectrum story and that the claim map was previously stale, so I’m treating the current docs as the authority and checking whether g185v2 overextends g186.

The prereg itself already exposes the core weakness: g185v2 scores a selector, but its comparisons mix different probe costs and may let a dose prior masquerade as geometry. I’m checking the current result artifact and the frozen export implementation before ranking attacks.

1. **S10: Ridge may be an alpha-prior, not a dose selector.** Current partial g186 already has Qwen3 best at `alpha=1.0` for 6/6 seeds; GPT-2 dose rows are not there yet. If the full run has one dominant dose, frozen Ridge can “work” by decoding KD strength from geometry and always selecting the population optimum.

2. **S10: g186 PASS could be meaningless for g185v2.** g186 can pass if geometry predicts the average dose-response curve, not the per-seed best dose. That supports “geometry tracks intervention magnitude,” not “geometry enables actionable seed-level compute savings.”

3. **S9: quality retention is malformed.** `mean_NLL_geometry / mean_NLL_oracle >= 85%` is too easy because lower NLL is better and ratios near 1 pass even when the selected run is worse. Use regret or improvement retention: `(scratch_NLL - selected_NLL) / (scratch_NLL - oracle_NLL)`.

4. **S8: export risk.** `export_frozen_ridge()` will export with `len(delta_X) >= 20`; right now that could mean a mostly/all-Qwen Ridge before g186 is complete. The prereg says freeze only after g186 PASS, but the code does not enforce that.

5. **S8: compute-savings claim is unfairly framed.** Geometry probes all 5 doses to 3%; alpha heuristic probes 1. If geometry only ties alpha=1.0, it spends more compute for no value. “80% savings vs brute force” is not enough; it must beat the cheapest strong prior in regret per FLOP.

6. **Clean falsifier:** run g185v2 as counterfactual all-60 completion, but score a frozen `geometry Ridge` against a frozen `alpha-only / arch+alpha / population-best` selector trained on the same g186 rows. Claim survives only if geometry has lower paired regret than those priors on fresh seeds, seed-block CI excluding zero, especially on cases where oracle best is not the g186 population mode. If it selects `alpha=1.0` >=80% or fails that regret test, kill the claim.  
<oai-mem-citation>
<citation_entries>
MEMORY.md:295-329|note=[prior moonshot-llm-genome context and narrowed forecast claim]
</citation_entries>
<rollout_ids>
019db969-52ed-7ea2-8a05-5e42846ceb8c
</rollout_ids>
</oai-mem-citation>

I’ll ground this in the repo files you named, then answer as a hostile launch review rather than a general strategy note. I’m also doing a quick memory pass because this repo has prior moonshot context that affects the claims framing.

The current registry already marks g186 as the live causal-intervention test and g185v2 as draft gated on g186 PASS. I’m reading the preregs and the Ridge export code now to find the pre-launch failure mode.

The failure shape is already visible: g185v2 assumes a deployable selector, while g186 only needs statistical prediction of deltas. I’m checking whether current result artifacts already show the dose-prior or alpha=1.0 collapse directly.

**Kill g185v2 for now.**

1. **Severity 10: frozen Ridge can be a dose decoder, not a selector.** g186’s held-out-seed CV can pass if early geometry simply encodes KD alpha. Fresh seeds still share the same architectures, doses, corpus, teacher-text loss, and feature step, so “generalization” may just be recognizing the alpha-induced perturbation signature.

2. **Severity 10: alpha=1.0 may be the whole trick.** Current partial g186 artifact has only 34/60 cells, but Qwen3 already shows alpha=1.0 best on 6/6 seeds, with mean delta peaking at alpha=1.0. A frozen Ridge that always selects alpha=1.0 can look “prospective” while adding nothing over the preregistered alpha heuristic.

3. **Severity 9: export_frozen_ridge is too permissive.** It trains on all available rows, allows export at 20 rows, does not require g186 PASS, does not require the conditioned permutation, does not reject alpha-decodability, and reports in-sample training R2. That artifact is not a scientifically locked control object.

4. **Severity 9: alternative g186-PASS explanation kills g185v2.** g186 may show a population dose-response curve: geometry tracks the global KD perturbation curve, not seed-specific intervention susceptibility. Then g185v2 is meaningless because the actionable policy is “use the population-best alpha,” not “measure geometry.”

5. **Severity 8: 85% quality retention is too easy and probably mis-specified.** With NLL, `mean_NLL_geometry / mean_NLL_oracle` is backward: worse NLL gives a larger ratio. The real metric should be regret or improvement retention: `(scratch - selected) / (scratch - oracle)`, plus a tight regret bound.

**Resolving experiment:** before g185v2 launches, run an offline locked replay on completed g186: freeze Ridge, alpha-only, and population-mean policies; require selection entropy, <80% agreement with alpha=1.0, lower regret than alpha=1.0 and population-mean, and improvement-retention >=85% using the corrected formula. If Ridge mostly picks alpha=1.0, archive g185v2.