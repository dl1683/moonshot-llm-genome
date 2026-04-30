Read the requested docs plus the g183 prereg. Local g183 result is still incomplete: only `scratch_ce seed=42` is written, so I’m treating the full run as unknown.

**1. If g183 Passes**
It means the +0.513 nat effect is not “donor capability transfer.” It is an interface prior: a useful token embedding/lm_head geometry held in place during SGD.

But PASS alone does **not** prove “token co-occurrence structure” is the active ingredient. It proves only “a corpus-built, non-random interface can help.”

Controls that distinguish it:

- `ppmi_svd_shuffled_rows`: if this matches PPMI, token identity is not load-bearing.
- `frequency_anchor`: if this matches, unigram/frequency calibration is enough.
- `random_structured_anchor` / spectral-matched random: if these match, it is mostly scale/spectrum/non-random conditioning.
- PPMI with norm/logit calibration matched to Qwen3: needed because smoke shows PPMI starts badly miscalibrated.
- Init-only vs continuous-anchor: separates “better starting point” from “persistent regularizer.”
- Corpus mismatch control: PPMI from shuffled text or different-domain corpus. If real C4 PPMI wins, corpus co-occurrence matters.

**2. If g183 Fails**
Then the +0.513 nat effect is a learned autoregressive interface prior, not raw corpus statistics.

What pretraining captures that PPMI misses:

- directional next-token conditional structure, not symmetric window co-occurrence;
- logit/softmax calibration and row norms;
- subword/tokenizer conventions: whitespace, continuations, rare-token smoothing, special-token behavior;
- polysemy and context-dependent usage collapsed by static type vectors;
- embedding/lm_head co-adaptation to hidden-state distributions;
- higher-order syntax and phrase structure beyond window-5 co-occurrence.

The smoke result already points this way: PPMI has semantic neighborhood structure, but starts with terrible NLL because it is not a calibrated LM interface.

**3. Competitive Scan**
No direct Apr 28-30 threat surfaced for “PPMI/SVD corpus-derived LLM initialization.” That remains open.

For KD: the Apr 28 paper [Knowledge Distillation Must Account for What It Loses](https://arxiv.org/abs/2604.25110) is not an alpha-prediction law, but it supports a richer KD compatibility target: predict not just score gain, but which teacher properties will be lost. Older ACL work, [Towards the Law of Capacity Gap](https://aclanthology.org/2025.acl-long.1097/), is relevant prior art: it predicts teacher scale from student scale, but not optimal KD alpha/dose.

For early geometry: no new Apr 28-30 direct hit beyond the already-dangerous April cluster: [Geometric Canary](https://arxiv.org/abs/2604.17698), [in-training LLM probes](https://arxiv.org/abs/2604.01025), [Spectral Lifecycle](https://arxiv.org/abs/2604.22778), and [ILDR grokking detection](https://arxiv.org/abs/2604.20923). The opportunity is now narrower: task-aligned diagnostics beat universal geometry.

**4. If g183 Fails: Direction Ranking**
1. **KD compatibility law**: best fireable path. g186 says alpha/arm identity dominates; convert that into a law predicting optimal alpha from teacher entropy, student capacity, tokenizer overlap, corpus mismatch, and CE/KL gradient ratio. Ceiling: ~6.3-6.8.
2. **g155 production efficiency**: highest eventual ceiling, but currently hardware-blocked. If wall-power measurement is unblocked, this can go back to ~8.0 because it is product-grade efficiency, not phenomenology.
3. **Completely different pivot**: only good if specific. Best version: “interface compatibility compiler” for small LMs: choose tokenizer, embedding/lm_head init, KD dose, and corpus mix to minimize NLL/kJ. Generic pivoting resets progress and likely stays <5 for 50 cycles.

**5. §0.1 Projection**
Realistic next-50-cycle ceiling: **~6.0**, maybe **6.5** if KD compatibility works cleanly.

A return to 7+ needs either g183 PASS with adversarial controls or g155 unblocked with real energy-efficiency proof. Current trajectory after g186 FAIL and PPMI smoke miscalibration does not support an 8/10 path without a concrete product-grade efficiency result.

Sources: arXiv/ACL links above; local grounding from `WIKI.md`, `research/CLAIM_EVIDENCE_MAP.md`, and `research/prereg/genome_183_corpus_derived_init_2026-04-30.md`.

