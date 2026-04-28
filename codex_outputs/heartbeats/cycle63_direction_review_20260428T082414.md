# Cycle 63 Direction Review

## 1. Sequencing

Pick **(d): g173 immediately after g177v2; prepare g178 CPU-side, but do not spend GPU until g173 verdict**. g173 is the only queued test that attacks A6: single-family trained-basin artifact. It is also the only one that can honestly move the ceiling to 8.0-8.4. g178 is valuable, but explanatory; before cross-arch survival it risks making a detailed map of a Qwen3-family basin. Cheap leverage is false if it delays the decisive 3.85h run. If g173 PASS/intermediate, run g178 to localize the donor-bearing subsystem. If g173 FAIL, redesign g178 as failure autopsy with alt-trained/cross-arch donors, not just attention/MLP decomposition.

## 2. Derivation Status

The best first derivation is **basin-of-attraction / anchored gradient flow**, not PAC-Bayes first. Write recipient training as `L_data(theta) + lambda/2 ||theta - theta_d||^2`, with random `theta_0`. Around the early training trajectory, diagonalize local curvature `H = U diag(h_i) U^T`, then project donor displacement `z = U^T(theta_d - theta_0)` and task gradient `g = U^T grad L(theta_0)`. The anchor gives mode-wise force `lambda z_i` and fixed-point displacement `a_i(lambda) = lambda/(h_i + lambda) z_i`. Predicted NLL gain should scale like `sum_i[g_i a_i - 1/2 h_i a_i^2]`: monotone at low lambda, saturating when lambda exceeds informative curvature, and reversing if high-curvature/non-task-aligned modes dominate.

Falsifiable <4h test: run 4-5 lambda values around g165 (`0.0003, 0.0013, 0.003, 0.01, 0.03`) on 1-2 seeds; estimate top Hessian/Fisher directions or gradient-donor cosine at step 0/50; predict the lambda response curve. Rate-distortion can later interpret useful `z_i` as code length. PAC-Bayes will likely be too loose.

## 3. Competitive Landscape

Likely 6-month threats:

- **Cross-architecture merging** becomes normal. LS-Merge and Transport-and-Merge already target heterogeneous weight transfer; scaled versions could subsume "donor structure crosses architecture."
- **Model-merging scaling laws** make g173 look like a small instance of an expert-count/model-size power law.
- **Weight-selection/subcloning/HyperCloning** recasts anchoring as pretrained initialization or training-speed transfer.
- **KD capacity-gap/reasoning-distillation** work leapfrogs C19/C20 with larger benchmarks and teacher-size laws.

Moat: not "KD helps" or "weights can merge." The moat is **random-init recipient + continuous donor constraint + matched nulls + kill criteria**, showing trained structure works as an active basin force during SGD, while zero-step transplant and decay fail. Big labs will optimize deployed transfer; they are less likely to publish adversarial negative controls saying pretrained weights are not capability unless optimization keeps tension on them.

External priors checked: Model Soups `arxiv.org/abs/2203.05482`; LS-Merge `openreview.net/forum?id=VSDV0SWwOC`; Transport and Merge `arxiv.org/abs/2602.05495`; Model Merging Scaling Laws `openreview.net/forum?id=vpKXTmMtBQ`; Weight Subcloning `arxiv.org/abs/2312.09299`; Capacity Gap `arxiv.org/abs/2311.07052`; Small Models Struggle `arxiv.org/abs/2502.12143`.
