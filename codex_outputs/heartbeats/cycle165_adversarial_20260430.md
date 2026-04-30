Verdict: the smoke makes **scalar-only** look weak, but it does **not** yet prove “trained embedding directions are the interface signal.” It proves “correct-token angular targets under a continuous tied embedding/head anchor are much less toxic than wrong angular targets” in a 50-step pilot.

**1. SEV-10: tied `lm_head` confound**
The current code uses tied word embeddings, and g191/g194 inject/anchor `model.embed_tokens.weight`; with tied weights, that is also the output classifier basis. The effect may be **output-logit class-vector prior**, not input embedding/interface geometry. See [g188 tie setup](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_188_tokenizer_flow_bridge.py:326>) and [g191 injection/anchor path](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_191_string_match_decomposition.py:181>).

Resolver: untie embeddings and run input-only, output-only, both, and crossed input/output direction arms. If output-only gets most of the gain, kill the “embedding interface” claim.

**2. SEV-10: continuous-anchor dominance**
g191 says anchor-only is ~98% of the signal, so this may be an active regularization/tether effect, not transferable row content. The docs already flag this: [CLAIM_EVIDENCE_MAP.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:68>).

Resolver: for every g194 arm, split `init_only`, `anchor_only`, `init+anchor`, and `anchor_cutoff_after_{50,500,2000}`. The claim survives only if correct directions leave residue after anchor removal.

**3. SEV-9: “direction” may mean well-conditioned angular scaffold, not trained semantic signal**
`correct_dir_uniform_norm` beating `full_match` is dangerous. It says the natural trained vector is not optimal here; stripping trained norms improves the pilot. That supports “some angular basis helps,” not necessarily “the trained embedding rows are the interface.”

Resolver: norm sweep around uniform/full/frequency norms, matched anchor-gradient norms, and token-level CE deltas. Require trained directions to beat random orthogonal/angular bases with the same conditioning.

**4. SEV-9: exact-string lookup table, not a transcodable law**
Mystery 8 now says exact lexical identity carries the signal; g193 failed to synthesize directions from byte/length/frequency features. That can mean the “signal” is an uncompressible lookup table over shared tokens, not a general interface law. See [OPEN_MYSTERIES.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/OPEN_MYSTERIES.md:174>).

Resolver: low-overlap tokenizers, semantic-neighbor transfers, contextual/distributional compilers, and held-out non-exact tokens. If exact string match is required, frame it as lexical row reuse, not interface geometry.

**5. SEV-8: shallow/horizon/token-mass artifact**
The smoke is 50 steps, 1 seed. The full g194 result file is still incomplete. Prior docs already require g192 because 8-layer/5000-step may not survive full 28-layer training: [WIKI.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:465>).

Resolver: finish g194, then g192 full-depth, longer horizon, balanced token-frequency eval, punctuation/whitespace-stripped eval, rare-token eval, and OOD text. If the gain is mostly high-frequency shared-token mass, the narrative shrinks hard.

Bottom line: do not promote beyond “direction-favorable pilot” until the tied-head and anchor-residue tests pass. Those are the two real kill shots.

