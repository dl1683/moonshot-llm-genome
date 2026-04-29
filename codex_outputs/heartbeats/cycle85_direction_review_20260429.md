1. **Q1:** If g182 reference-free geometry beats `arm_mean` and `combined_telemetry` in both held-out folds, honest score is **8.3-8.6/10**. If it only catches the BERT sign flip while telemetry ties/beats it, **6.5-7.0**.

2. **Q2:** Use **residualized treatment-effect geometry**: predict labels after subtracting train-fold `arm_mean`, using init→3% deltas, KD-minus-scratch geometry drift, spectral rank/anisotropy/entropy shifts, tokenizer fragmentation, embed-lm_head alignment, and teacher-student representation mismatch. Keep raw arm/protocol IDs out of the geometry model.

3. **Q3:** After g182 PASS, run a **prospective closed-loop triage** test on a new family, ideally SSM/hybrid: freeze model and thresholds at 3%, then choose kill/continue/switch-arm. Score real compute saved at equal final quality versus loss-only, `arm_mean`, and combined telemetry.

4. **Q4:** If g182 FAILS, stop selling Genome Forecast; pivot to a **Tokenizer/KD Compatibility Benchmark** that predicts when cross-tokenizer KD harms. Ceiling is roughly **4.0-4.5/10** unless a first-principles invariant or efficiency-grade result emerges.

5. **Q5:** Yes: [Fast and Accurate Probing of In-Training LLMs’ Downstream Performances](https://arxiv.org/abs/2604.01025), submitted April 1, 2026, is the direct competitor: internal-representation probes, OLMo3-7B checkpoints, AUROC > 0.75, eval latency ~1h to ~3min. Adjacent threat: [Neural Neural Scaling Laws](https://arxiv.org/abs/2601.19831); [DWA-KD](https://arxiv.org/abs/2602.21669) is more of a cross-tokenizer KD-method threat.

