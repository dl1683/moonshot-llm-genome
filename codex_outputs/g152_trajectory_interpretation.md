# g152 Status Report

Basis: [CLAUDE §0/§0.05/§0.1](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:9>), [WIKI g152/g156 block](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:1106>), [C12](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:27>), [transport theory](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/derivations/prefix_information_transport.md:7>), [post-g156 plan](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/programs/post_g156_pass_program.md:18>), [g152 raw JSON](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_152_long_horizon_crossover.json>), [prior kill audit](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/codex_outputs/adversarial_kill_arch_prior.md:57>).

Under [CLAUDE.md §0.1](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:50>), the honest current status is still: **not a breakthrough**. DeepMind could publish a cleaner version tomorrow if they wanted to. `g152` removed the clean “baseline overtakes” killshot inside the observed horizon, but what survived is a **small, attenuating, regime-dependent effect**, not a flagship claim. In the raw JSON, final top-1 is still positive at `+0.2667pp C4` and `+0.4526pp OOD`, but the final C4 NLL is slightly worse for minimal (`+0.0258` nats), and the 3-seed final top-1 intervals include zero.

## A) Theory status

`g152` **strengthens the prefix-information-transport theory a little and weakens the broad architecture-prior thesis a lot**.

The supportive part is real. The theory says MLP parameters are only wasted while the transport gap is open; once transport saturates with compute, the advantage should shrink. That is exactly the observed shape in [g152 raw JSON](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_152_long_horizon_crossover.json>): mean top-1 peaks at `+1.6037pp C4 / +1.7018pp OOD` at `(8000,16000)` and shrinks to `+0.2667pp / +0.4526pp` at `(25000,50000)`. That is a `6.0x` shrink on C4 and `3.8x` on OOD.

But this is **consistency evidence, not discrimination**. The same trajectory is also compatible with the banal story: “the smaller transport-heavy arm is more compute-efficient early, the bigger MLP arm catches up toward parity later.” So `g152` does **not** prove transport, and the derivation doc is too strong where it implies the artifactual reading is falsified. What `g152` really does is make `g156` more valuable, because `g156` is the orthogonal-axis test that can separate “transport control variable” from “early-budget artifact.”

So: **yes, `g152` is evidence for the theory in the weak sense that it matches the predicted attenuation shape; no, it is not evidence strong enough to upgrade the thesis on its own; yes, it strengthens the case for running `g156`.**

## B) Projection to larger scale

Descriptive correlations from the four averaged checkpoints in the JSON:

- Full 4-point Spearman ρ(`compute`,`gap`) = `-0.40` on C4, `-0.80` on OOD.
- Post-peak Spearman ρ on the last 3 points = `-1.0` on both, but that is only 3 points, so treat it as descriptive shape, not strong inference.

Fit quality on the **post-peak** attenuation only:

- Best fit among the requested forms is **exponential decay**.
- Most conservative extrapolation is **power-law decay**.
- Global linear decay is not credible long-range; it goes absurdly negative. As a short-range local extrapolation only, it predicts the gap hits zero around `1.23x` final compute on C4 and `1.32x` on OOD.

Conservative power-law projections, in percentage points:

| Scale proxy | C4 gap | OOD gap |
|---|---:|---:|
| `1B` rough proxy (`~5x` current compute) | `0.025pp` | `0.080pp` |
| `7B` rough proxy (`~35x`) | `0.0012pp` | `0.0088pp` |
| `70B` rough proxy (`~350x`) | `0.00003pp` | `0.00065pp` |
| `10x` final compute | `0.0084pp` | `0.036pp` |
| `100x` final compute | `0.00024pp` | `0.0027pp` |
| `1000x` final compute | `0.0000069pp` | `0.00020pp` |

The exponential fit collapses even faster and is effectively zero by `10x`.

Statistical distinguishability: **no**. Not at `1B`, not at `7B`, not at `70B`, not at `10x/100x/1000x`. On the present 3-seed noise floor, the final checkpoint is already not cleanly distinguishable from zero:

- Final C4 paired-gap 95% CI: `[-0.42, +0.95]pp`
- Final OOD paired-gap 95% CI: `[-0.06, +0.97]pp`

Raw final seed gaps make that plain:

- C4: `-0.04`, `+0.35`, `+0.49` pp
- OOD: `+0.26`, `+0.43`, `+0.67` pp

The hard conclusion is: **if you extrapolate honestly, the long-run gap goes to practical zero.**

## C) Reframing for §0.1

Does attenuation nuke the thesis? **It nukes the strong version.** You can no longer honestly sell “MLP/depth are broadly wasted” or “scale strengthens the win.” `g152` says the opposite: the effect decays toward the noise floor as compute increases.

Does attenuation plus transport theory produce a stronger claim? **Only conditionally.** If `g156` PASSes, the stronger claim is not “minimal wins forever”; it is: **transport demand is the control variable, and the `g152` attenuation was an ex ante predicted saturation pattern.** That is materially better than a narrow ablation paper because it moves from phenomenology toward derivation. But `g156` PASS alone still does not complete the story; the locked post-`g156` program explicitly says the real claim only arrives after `g157/g158/g159/g161/g160` PASS.

§0.1 score:

| State | Score | Honest reading |
|---|---:|---|
| Now, post-`g152` | `4/10` | Narrow regime-specific empirical effect; not breakthrough |
| With `g156 PASS` | `6/10` | Serious theory lead, still not closed out as design law |
| With `g156 KILL` | `1/10` | Breakthrough-axis dead; demote to low-budget family-specific curiosity |

## D) Decision

**Pick: (i) Continue exactly as planned (`g156` next).**

Reason: within the remaining `~1hr`, `g156` is the **only** live action that advances [CLAUDE.md §0.1](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:54>) on axis `(a)` or `(b)`. More attenuation work or `g153` is the patch-old-chain trap, and the locked post-`g156` plan explicitly says `g152`/`g153` are **not** in the breakthrough stack because they “patch the old empirical chain rather than add new evidence types” [post_g156_pass_program.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/programs/post_g156_pass_program.md:26>). Immediate pivot to distillation before `g156` resolves would abandon the only still-live derivation route for a one-hour cost.

Operationally:

- Let `g156` finish.
- If `PASS`, queue `g157` immediately, then stay on the locked transport program.
- If `KILL`, stop treating architecture-prior as a breakthrough axis and pivot hard to `g154 → g155`.

That is the only decision consistent with the manifesto end-goal: **either turn transport into a law that can guide capability transfer / student design, or kill it and move directly to efficiency productization.**

## E) One-sentence headline

As of `g152`, the no-MLP minimal arm still stays ahead through the full 200M matched-budget horizon, but its advantage decays from a `+1.60/+1.70pp` mid-horizon peak to a final `+0.27pp C4 / +0.45pp OOD`, so architecture-prior currently stands as a small, attenuating, regime-dependent effect consistent with, but not yet validating, the prefix-information-transport hypothesis.