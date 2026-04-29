Q1: No. This confirms A11 arm_mean failure, not §0.1=6.6-7.0 yet. The frozen g180 geometry model also appears wrong on current g180b rows, so score is ~5.8-6.2: important phenomenon, not validated diagnostic. ≥7 needs g182 Model B beating arm_mean and combined telemetry.

Q2: Keep g182, but treat `seq_kd_full` as a likely failure arm, not a transfer arm. GPT-2 should probably show the same KD-hurts pattern, maybe worse, because it adds real architecture mismatch to tokenizer mismatch. Do not add DWA-KD into g182; make it g184/control after g182.

Q3: Theory: tokenizers define incompatible probability spaces. Teacher outputs are segmented, entropied, and length-shaped for Qwen’s vocabulary; BERT/T5/GPT-2 impose different boundaries, EOS behavior, token frequencies, and many-to-many vocabulary mappings. Vocab size matters, but BERT/T5 both hurting says sequence/semantic alignment is the deeper issue.

Q4: Yes, DWA-KD directly supports this interpretation. The arXiv abstract says cross-tokenizer KD is limited by poor sequence/vocabulary alignment, and DWA-KD uses entropy weighting plus Soft-DTW over embeddings/final hidden states to repair that. It explains why naive CTKD can be harmful, though not your exact sequence-KD setup. Source: https://arxiv.org/abs/2602.21669

Q5 queue score:
- Finish g180b: 7/10 as falsifier/phenomenology, not forecast proof.
- g182: 9/10. Highest leverage; unchanged.
- g183 corpus init: 5/10 until g182 passes.
- g182 phase-2 SSM: 8.5/10 only after phase-1 pass.
- New g184 DWA-KD rescue/control: 7/10, competitive-context follow-up.

