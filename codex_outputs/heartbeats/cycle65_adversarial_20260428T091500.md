# Cycle 65 Adversarial Review

## Top 3 attacks ranked by severity

1. Tokenizer/embedding artifact — severity 9/10. Every surviving positive result lives in Qwen3 token space: g165 anchors full Qwen3 weights, g167 KD uses Qwen3 vocab/logits, and g174/g177v2 nulls do not remove the tokenizer/embedding/head surface. Because the 151k-token embedding/readout block is a huge part of the optimization geometry, the +1 nat may be mostly "trained Qwen-token lexical prior" rather than transferable internal structure. g177v2 actually strengthens this objection: unrelated same-arch C4 donors recover 95-96% of Qwen3's effect.

2. 500-step horizon artifact — severity 8/10. g165/g174 weight-anchor evidence is measured at 500 steps, exactly where a continuous anchor can look like capability transfer while scratch has not yet had enough CE updates to discover the same basin. The project already saw long-horizon attenuation in g152, and g172 says KD is strongest late, not at initialization. Until scratch/weak-anchor/anchor-release are tracked to 5000+ steps, the honest claim is "short-horizon acceleration," not persistent basin attraction.

3. Narrow protocol basin: lambda plus recipient seeds — severity 7/10. The locked effect is at lambda=0.01, where the anchor gradient was already estimated around 7.6x CE at step 0; larger lambdas were excluded as likely donor-clone collapse rather than tested. The seed evidence is also only recipient seeds [42,7,13], reused across g165/g167/g174/g177v2, so tight paired CIs may understate protocol dependence. A real trained-structure law should survive broader recipient seeds and show a smooth dose-response, not only one tuned regularization strength.

## Single resolving experiment

Run g165-style Part A to 5000 steps with seeds [42,7,13,101,202,303]. Arms: scratch; full_anchor lambda {0.003,0.01,0.03}; full_anchor lambda=0.01 released after step 500; embed+lm_head_only anchor at gradient-matched lambda; no_embed_lm_head anchor at gradient-matched lambda. Same C4 train/eval protocol, report step 500 and 5000. Claim survives only if no_embed_lm_head and full_anchor both retain >=+0.5 nats vs scratch at 5000, beat embed-only and release arms by >=+0.3 nats, and are positive in >=5/6 seeds. Otherwise call it tokenizer/short-horizon/regularization artifact.

