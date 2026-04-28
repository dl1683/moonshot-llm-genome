# Cycle 60 Adversarial Review

## Top 3 attacks ranked by severity

1. NLL-extrapolated donor identity is assumption-dominated — severity 9/10. g177v2 cannot cleanly re-lock C22 if the only alt-donor points sit at NLL ~5.7-5.8 while Qwen3 is at ~3.55. A 95% PI extrapolated ~2.1 nats outside a 3-point cluster is mostly a linearity prior, not empirical exclusion. A critic can say the experiment proves only that weak undertrained donors transfer less, not that Qwen3 identity has a special residual.

2. Same-family trained-basin artifact — severity 8/10. C18 and C19 are still one Qwen3-shaped recipient, one C4-like task, and one canonical donor family/strength setting. C21 kills random/permuted/uniform nulls, but it does not kill “any sufficiently trained Qwen3-shape model on similar data induces the same basin attraction.” The live claim should be “trained-structure-specific continuous constraint works in this family,” not “neural genome transfer” in the strong sense.

3. C20 matched-compute null gap — severity 7/10. Late KD’s +0.700 pp at 33% KD compute is compared against same-step scratch, not against scratch given the same extra non-KD compute or a late CE-only fine-tune schedule. If scratch+2000 extra CE steps, LR-reset CE, or matched wall-clock late training recovers comparable top-1, the “compute-efficient transfer” framing collapses into ordinary late-training marginal gain. Minimum kill: run g172b scratch_8000 / late_CE_4001-6000 / late_KD_4001-6000 with the same seeds.

## Single resolving experiment

Run g177b trained-alt-basin control: alt donors seeds 1234/5678/9999, same Qwen3 architecture, same C4/dedup stream, save checkpoints at 10k and 30k steps; recipient seeds 42/7/13. Arms: scratch, anchor_qwen3, anchor_alt_seed_checkpoint, kd_qwen3, kd_alt_seed_checkpoint. Criteria: C18/C19/C21 strong claim dies if any alt checkpoint reaches >=80% of Qwen3 gain or Qwen3-minus-best-alt <=0.2 nats / <=0.2 pp with CI crossing zero. C22 stays provisional unless alt checkpoints span at least 1 nat of donor NLL and Qwen3 remains above the trend PI.