**Ranked Kill List**

1. **SEV-10: Arm identity can masquerade as geometry.** Arms are visible by step 108, and C' uses spectral/rank/drift/ID/kNN features that can easily encode “seq_kd_full vs embed_anchor,” not health. The prereg includes arm-label and arm-mean baselines, but a C' win still may be an arm classifier with smoother continuous proxies. This is fatal if C' predicts arm above chance or loses after within-arm residualization. See `WIKI.md:33`, `research/prereg/...:78-83`, `code/genome_182_triage_arena.py:577-586`.

2. **SEV-9: Overfit risk is underpriced.** C' is 8 correlated features on about 24 labeled training cells per LOAO fold, with only 12 seed blocks and Ridge alpha selected inside that fold. Ridge helps, but does not fix adaptive feature/model selection after repeated adversarial cycles. A few high-leverage seeds can create a clean-looking cross-arch point estimate.

3. **SEV-8: KS “alignment” test is weak evidence.** P3 uses two-sample KS within arm, so n=12 vs n=12. At this size, KS only detects very large shifts: D critical is roughly 0.50-0.56. Therefore “p > 0.1 for 6/8 features” is non-rejection, not equivalence. It can miss moderate architecture-specific feature distributions while still letting Route 3 survive. See `research/derivations/...:101`, `code/genome_182_triage_arena.py:2011-2035`.

4. **SEV-7: Alternative explanation remains simpler.** Qwen3 and GPT-2 are both English decoder Transformers on C4; GPT-2 was also the only g180b tokenizer where geometry helped. The C' signal may be “nearby tokenizer/interface/protocol transfer,” not architecture-independent manifold health. The prereg itself admits GPT-2 only supports “cross-Transformer-family,” not full non-attention generality.

**Resolving Experiment**

Run one frozen, no-refit **arm-residual OOD kill** on Falcon-H1: train C' on Qwen3+GPT-2 using labels residualized by arm mean and features residualized against arm one-hot; freeze scaler, alpha, coefficients. Test on Falcon-H1 within-arm residual outcomes only. PASS requires C' beating telemetry and arm-residual baselines with seed-block CI > 0; FAIL if arm can be decoded from C' features or residual MSE gain <15%. This directly kills the arm/protocol explanation.