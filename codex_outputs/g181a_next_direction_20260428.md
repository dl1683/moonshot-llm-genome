# g181a Next Direction

Candid read: the transfer-mechanism story is dead in the strong form. g181a is not scientifically "intermediate": embed_lm_head_only gives +0.483 nats, no_embed_lm_head gives -0.439 nats, and no_embed - embed = -0.923 nats [CI -1.055, -0.835]. Combined with C22 rejection and the g173 locked-criterion fail, C18/C19/C21 should be reframed as Qwen3-tokenizer/embed prior and family-scoped KD effects, not neural genome transfer of internal structure.

Scores by section 0.1 uplift potential:

1. g180 Genome Forecast original: 8.0/10. Still high, but only if stripped of transfer-language and judged as early run-quality prediction versus early-loss baselines.
2. g181b long-horizon attenuation: 3.0/10. Worth doing only as audit hygiene. If PASS, it says tokenizer prior persists; if FAIL, it says short-horizon acceleration. Neither revives the moonshot.
3. g182 zero-step-tokenizer-transplant: 5.5/10. Good kill/clarify experiment. If +0.5 appears at step zero, continuous SGD dies too; if not, there is still an "init prior must be maintained during optimization" mechanism. But it remains tokenizer-prior, not genome transfer.
4. Pivot Genome Forecast: early tokenizer/embed geometry bad-run diagnostic: 9.0/10. Highest leverage. It uses the negative result as the discovery: early vocabulary/embedding geometry may be a cheap predictor of downstream trainability, tokenizer fit, and doomed-run risk. This can become a practitioner tool.
5. Non-obvious: tokenizer-prior compatibility benchmark across tokenizers/corpora: 7.5/10. Measure whether a step-0 geometry score predicts which tokenizer/init/corpus combinations train efficiently before spending GPU. Strong product potential, but narrower than forecast unless tied to run-level decisions.

Pick: candidate 4. Do not run g181b as mainline; it is now a cleanup experiment. Do not run g182 as mainline unless the next question is specifically "is continuous SGD dead too?" The highest section 0.1 move is to pivot the headline to Forecast/Diagnostic:

"Can we predict final run failure or compute efficiency from zero-to-3-percent tokenizer/embed geometry better than early loss?"

Why this is powerful: it is cheap, falsifiable, cross-arch from day one, and turns the g181a damage into a mechanism. The new story is not "we transfer learned structure between models." It is "the earliest geometry of the token/embedding interface reveals whether training will be healthy, wasteful, or doomed." That is still moonshot-grade if it predicts bad runs before expensive training, because it creates a training triage instrument.

Required bar: held-out runs, multiple tokenizers and architectures, baseline against early validation loss, AUROC for bad-run risk, and simulated compute-savings policy. If geometry does not beat early loss alone, pivot lower. If it does, this is the new flagship.
