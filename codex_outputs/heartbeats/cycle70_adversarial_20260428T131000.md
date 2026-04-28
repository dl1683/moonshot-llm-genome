# Cycle 70 Adversarial Review (g173 INTERMEDIATE reframe)

## Top 3 attacks ranked by severity

1. Methodology drift from locked endpoint — severity 10/10. The preregistered PASS criterion was final-accuracy ratio >=1.5x, and it failed. Recasting the result around gain ratio after seeing the table is exactly the post-hoc metric switch a skeptical reviewer is trained to reject. The 2.86x number may be interesting, but externally it has to be labeled exploratory unless reproduced under a fresh preregistered endpoint.

2. Underpowered near-chance benchmark delta — severity 9/10. The headline lift is only +2.29pp over a 39.87% scratch baseline, with component tasks sitting at or barely above chance under N=500 evals. At that floor, seed variance, item sampling, and benchmark discreteness can easily manufacture 0.5-2pp movement. With only n=3 seeds and no paired CIs, the reviewer can reasonably assume most arm-vs-arm contrasts cross zero until proven otherwise.

3. Architecture interpretation is confounded — severity 8/10. The Llama student is 173M while the Qwen-arch student is 596M, so "Llama gets more KD lift" may just mean the smaller model has more short-horizon headroom at 3600 steps. Worse, both students share the Qwen3 tokenizer, and prior tokenizer-init evidence suggests trained vocabulary/interface effects can dominate early anchor gains. That makes the result look less like architecture-crossing representation transfer and more like tokenizer-mediated initialization plus capacity/training-horizon confounding.

## Verdict on the reframe

The "cross-arch GAIN ratio 2.86x" framing is not defensible as an external PASS claim. It is a useful exploratory observation, but the locked final-accuracy ratio failed, so presenting the gain ratio as the right metric after the fact reads as methodology drift. The honest claim is: "The preregistered criterion failed; a post-hoc gain-normalized analysis suggests Llama may benefit more from KD than Qwen-arch under this short-horizon setup." That is hypothesis-generating, not confirmatory.

## Single resolving experiment

Run a fresh preregistered replication with gain ratio declared primary before training. Match or bracket parameter count, include same-tokenizer and native-tokenizer arms, use >=10 paired seeds, report paired t and bootstrap CIs, and evaluate on larger fixed subsets or full validation where possible. The reframe survives only if Llama's KD gain remains materially larger than Qwen-arch after parameter count, tokenizer, and seed uncertainty are controlled.
