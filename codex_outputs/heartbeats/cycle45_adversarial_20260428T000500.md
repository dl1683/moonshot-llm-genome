**Strongest attack: donor-specificity is unproven. Score: 8/10.**

The data do not yet prove “continuous donor-information-in-the-loss.” They prove “continuous extra constraint/supervision improves this setup.” g165 lacks a matched null anchor: random/untrained/permuted/spectrum-matched Qwen weights at identical Frobenius distance and gradient norm. g167 lacks matched null KD: uniform, marginal-token, shuffled-top-k, or random-teacher logits at identical KL magnitude. Without those, “donor information” is an interpretation, not an isolated variable.

Alternative story: g165 is just a strong continuous regularizer that slows/reshapes early training and prevents bad basin drift. g167 is just dense soft-target supervision/label smoothing/sample-efficiency, not capability transfer. The fact that g170 says uniform KD beats transport-gated KD makes this worse: the active ingredient may be generic continuous supervision, not the teacher’s structured capability.

Decay failure is also overclaimed. g165’s decays are extremely fast relative to the 500-step run: step cutoff at 25, linear gone by 50, exp tau=10. That mostly tests “brief warmup,” not calibrated annealing. Slow decay to step 250/500, cosine, inverse-sqrt, or plateau-then-decay remains untested.

“Two independent mechanisms” is too strong. Both are Qwen3-0.6B anchored, same C4 distribution, same tokenization/eval regime, same small-recipient training horizon. g165 is same-architecture weight-space; g167 is Qwen teacher into Qwen-vocab minimal_3L. This is not generality. Real generality needs at least two teacher families, two student families, one non-C4 domain, and an end-task eval.

Magnitude is also vulnerable. g167 moves C4 top-1 from ~15.66% to ~16.67% while teacher is ~35.18%, closing only about 5% of the teacher-student top-1 gap. NLL gain is ~0.15 against a ~2.58 scratch-teacher gap. That may translate to negligible HellaSwag/C3_macro lift.

**Single falsifying experiment:** run one matched-control factorial: scratch, true Qwen anchor/KD, random-init Qwen anchor, permuted/spectrum-matched Qwen anchor, unrelated trained-teacher anchor/KD, uniform/marginal/shuffled-logit KD. Match compute, seeds, gradient norms, KL/Frobenius magnitudes. Kill criterion: if best non-donor null gets ≥80% of g165/g167 gain, or true donor beats best null by CI-crossing/no practical margin, the claim dies.