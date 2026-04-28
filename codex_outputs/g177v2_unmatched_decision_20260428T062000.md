Q1: Partly defensible, but the adversarial counter is valid.

Lambda-normalization satisfies the literal cycle 55 OR-clause if that clause truly allowed either NLL/Frobenius matching or per-donor lambda normalization. It is defensible procedurally: you did not fake parity, and you made the computational constraint explicit.

But scientifically, lambda-normalization only matches initial anchor-force scale. It does not make an undertrained donor equivalent to Qwen3. A donor at NLL ~5.3 plausibly contains less learned structure than one at ~3.55, so any weaker or different downstream effect can be attacked as “undertrained donor,” not “donor identity.”

Q2: Best defensibility per hour is **(b), with (c) as the framing**.

Run main cells with unmatched donors, but report donor NLL/Frobenius/lambda and plot `alt donor NLL × Δ`. The key defense is not “these donors are matched,” but:

“Because strict NLL parity is infeasible on this hardware, we treat alt donors as matched-corpus, force-normalized sensitivity probes. We explicitly test whether the effect tracks donor training quality. If Δ is not explained by NLL/Frobenius, residual donor-specific structure remains plausible.”

Do not spend 30k+ steps on one donor unless the scatter shows NLL is the dominant confound and you need a rescue experiment. One better-trained donor weakens n=3 inference and still may not reach parity.

So: keep `--allow-unmatched-donors`, but downgrade the claim. Make the unmatched donor analysis a confound-decomposition/sensitivity result, not a fully matched donor-parity experiment.