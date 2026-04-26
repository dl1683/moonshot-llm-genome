# Architecture-Prior Kill Report

The thesis, as written, is not currently defendable. The record supports a much narrower claim: at short training horizons, on one fixed C4/WikiText slice, a cheaper no-MLP Llama variant can be competitive or better. It does **not** yet justify “MLP and excess depth are non-essential” in the strong matched-FLOPs, capability-grade, scale-monotonic sense.

## Top 3 Strongest Attacks

1. **Short-horizon compute-optimality artifact**
   - **Severity:** 10/10
   - The repo itself identifies this as the strongest remaining attack in [genome_152_long_horizon_crossover.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_152_long_horizon_crossover.py:5>) and [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:1102>). The g152 log already shows the 200M baseline still climbing hard at longer horizons: 18.74% at 4k, 20.66% at 8k, 23.99% at 16k on C4 top-1 in [genome_152_run.log](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_152_run.log:12>). That is exactly what a late-crossover threat looks like.
   - If baseline catches or passes minimal later, the thesis collapses from “MLP/depth are wasted compute” to “smaller models are better in the low-budget regime.”
   - **Diagnostic experiment:** finish g152 exactly as specified: 200M, 14L+MLP vs 7L noMLP, 25k/50k matched-compute checkpoints, `N_TRAIN=131072`, 3 seeds, final decision on both C4 and OOD.
   - **Cost:** already scoped at ~3.3 GPU-hours on the current setup.

2. **The 30M matched-FLOPs replication claim is false on the face of the record**
   - **Severity:** 9/10
   - [CLAIM_EVIDENCE_MAP.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:26>) says C11 is “equal training FLOPs at 30M.” But g141 uses the same `TRAIN_STEPS = 4000` for both arms in [genome_141_minimal_prior_capability.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_141_minimal_prior_capability.py:54>), and the wiki itself labels the 30M point as **matched steps**, not matched FLOPs, in [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:1016>) and [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:1041>).
   - So the defended headline “matched-FLOPs at 30M, 100M, 200M” is internally inconsistent. One of the three anchor points is a different regime.
   - **Diagnostic experiment:** run a true 30M confirmatory matched-FLOPs test with explicit FLOP accounting, not step matching, on a fresh split.
   - **Cost:** roughly 1 hour of GPU time for 3 seeds.

3. **The confirmatory and capability-grade story is weak**
   - **Severity:** 8.5/10
   - C10-C12 have **no locked preregistration**, and C13’s “prereg” is actually a later mechanism follow-up file, not a prereg of g151, in [CLAIM_EVIDENCE_MAP.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:25>) and [CLAIM_EVIDENCE_MAP.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:28>). g151 is also a **single-seed** best-vs-best sweep over 4 LRs on the same eval set in [genome_151_arm_specific_lr.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_151_arm_specific_lr.py:61>) and [genome_151_arm_specific_lr.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_151_arm_specific_lr.py:68>).
   - The HellaSwag claim is especially weak. The test uses only `N_HELLASWAG = 500` in [genome_148_hellaswag_capability.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_148_hellaswag_capability.py:61>), both models are basically at chance, and the seedwise gaps are inconsistent: baseline 26.0/23.4/25.6 vs minimal 25.2/26.8/25.2 in [genome_148_run.log](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_148_run.log:17>) and [genome_148_run.log](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_148_run.log:69>). The mean +0.73pp gap in [genome_148_run.log](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_148_run.log:90>) is noise-level.
   - The HellaSwag scoring is also non-canonical: context and ending are tokenized separately and concatenated in [genome_148_hellaswag_capability.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_148_hellaswag_capability.py:125>), which can distort boundary tokenization.
   - **Diagnostic experiment:** preregistered confirmatory rerun with separate validation/test splits, 3-5 seeds, LR chosen on validation only, full HellaSwag validation with correct full-string tokenization.
   - **Cost:** about 0.5-1 GPU-day.

## Other Audit Hits

- I did **not** find a hidden implementation asymmetry that secretly favors the minimal arm. Same `SEQ_LEN=256`, same `attn_implementation="eager"`, same tokenizer/vocab across arms in the main scripts.
- The “100M/200M are single-seed” attack is wrong. g146 and g147 use `SEEDS = [42, 7, 13]` in [genome_146_matched_flops_bigdata_100m.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_146_matched_flops_bigdata_100m.py:53>) and [genome_147_matched_flops_200m.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_147_matched_flops_200m.py:52>). The single-seed weakness lands on g151.
- The C4 “held-out” slice is just the next chunk from the same shuffled `allenai/c4` **train** stream in [stimulus_banks.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/stimulus_banks.py:76>) and [genome_141_minimal_prior_capability.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_141_minimal_prior_capability.py:192>). There is no dedup or overlap audit in code. I could not verify overlap offline because the dataset is streamed from HF and network is blocked.
- The Wikitext OOD eval uses the Wikitext **train** split, not validation/test, in [genome_141_minimal_prior_capability.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_141_minimal_prior_capability.py:134>) and the later matched-FLOPs scripts. That is not leakage against C4 training, but it is sloppier than “held-out OOD” wording suggests.

## Novelty Challenge

“MLP-free” is **not** novel. Broadly, this territory is crowded: [Hyena](https://arxiv.org/abs/2302.10866), [RWKV](https://arxiv.org/abs/2305.13048), [Mamba](https://arxiv.org/abs/2312.00752), [RetNet](https://arxiv.org/abs/2307.08621), [FNet](https://arxiv.org/abs/2105.03824), [gMLP](https://arxiv.org/abs/2105.08050), [AFT](https://arxiv.org/abs/2105.14103), [Synthesizer](https://arxiv.org/abs/2005.00743), [Linear Transformers](https://arxiv.org/abs/2006.16236), and [Pretraining Without Attention](https://arxiv.org/abs/2212.10544) all publish some version of “standard Transformer components are not sacred.”

What may still be narrow-novel is the exact **same-family** claim: within a Llama-style causal decoder, keep attention/width/residuals, delete MLPs, reduce depth, and stay competitive under a fixed small-budget regime. I did **not** find a primary-source paper making that exact claim. But that novelty is thin, and the win size is small.

Would DeepMind publish this tomorrow if they wanted to? **Yes.** A stronger version, with exact FLOP accounting, 1-7B scale, real downstream tasks, and locked confirmatory design, is completely publishable by a big lab. So this fails the project’s own §0.1 “big labs can’t / won’t publish it” bar.

## The One Kill Experiment

Run the g152 falsifier and let it decide the thesis.

- **System:** 200M Llama-family only, same two arms as g147/g151.
- **Scale:** baseline 25k steps vs minimal 50k steps, `N_TRAIN=131072`, 3 seeds.
- **Measurement:** C4 top-1/NLL and WikiText top-1/NLL at matched-compute checkpoints; full HellaSwag only at final; LR fixed from a separate validation split, not chosen on test.
- **Falsifier:** if baseline is ahead by >0.3pp on both C4 and OOD at the final matched-compute checkpoint, the architecture-prior thesis is dead. It becomes a low-budget efficiency artifact, not a structural statement about MLP/depth non-essentiality.

## What To Do Next

The **top-1 attack to defeat next is the long-horizon crossover**. Nothing else matters until that is resolved. If g152 fails, stop writing this up as architecture-prior theory. If it passes, then clean up the regime-mixing and confirmatory weaknesses.

## Final Verdict

**Not worth publishing as a flagship claim now.** It is too exploratory, too regime-mixed, and too dependent on an unresolved late-crossover threat.

**If** you patch the top-1 attack and it survives, this becomes publishable as a **narrow efficiency ablation paper**, not a breakthrough:
- **Competitive-reality / §0.1 score:** 4/10
- **End-goal / capability-transfer score:** 2/10

The self-flattery risk is high. The result may be real, but the current thesis is larger than the evidence.