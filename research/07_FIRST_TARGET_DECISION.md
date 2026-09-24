# First Target Decision

## Decision

The first mechanism card should target:

> truth-versus-user-agreement conflict in sycophancy and motivated reasoning.

Working card ID: `MC-001`.

## Why This Target

The first target should be narrow, important, testable, controllable, and easy to falsify. Sycophancy under wrong user hints is a good first target because the behavior can be measured with objective tasks where the correct answer is known before generation.

This avoids the largest problem with hallucination as a first card: "unknown answer" and "unfamiliar entity" are easily confounded. Hallucination remains important, but it is a harder first card.

## Decision Matrix

Scores are 1 to 5. Higher is better.

| Candidate | Importance | Label Clarity | Tooling Fit | Intervention Plausibility | Null Strength | Market Relevance | Clean Failure Chance | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sycophancy / truth conflict | 5 | 5 | 4 | 4 | 5 | 5 | 5 | 33 |
| Hallucination gate | 5 | 3 | 4 | 4 | 3 | 5 | 4 | 28 |
| Evaluation awareness | 5 | 2 | 3 | 3 | 2 | 5 | 3 | 23 |
| Refusal-coherence tension | 4 | 3 | 4 | 4 | 3 | 5 | 3 | 26 |
| Planning-ahead | 3 | 4 | 4 | 4 | 4 | 3 | 4 | 26 |
| Cross-lingual concept mediation | 3 | 4 | 5 | 4 | 4 | 3 | 4 | 27 |

## Chosen Behavior

The model receives an objective question plus a user hint. The hint may be correct, incorrect, or absent. The target behavior is whether the model follows the truth or conforms to the user's incorrect hint.

Examples:

- arithmetic and symbolic reasoning;
- factual questions with unambiguous answers;
- multiple-choice science or history questions;
- simple code-output prediction;
- logic puzzles with a known answer.

The key contrast:

- no hint;
- correct hint;
- incorrect hint with mild confidence;
- incorrect hint with strong confidence;
- incorrect hint with social pressure.

## Candidate Internal Signatures

Possible signatures:

- a truth-checking or answer-confidence direction;
- a user-agreement or social-compliance direction;
- a conflict feature when internal answer and user hint disagree;
- attribution path from hint tokens to answer token;
- SAE features that activate on agreement, uncertainty, correction, or contradiction;
- hidden-state early warning that predicts whether the model will cave before it does.

## Intervention Hypotheses

Hypothesis A:

Suppressing user-agreement features or directions reduces wrong-hint compliance.

Hypothesis B:

Amplifying truth-checking or answer-confidence features increases correctness under wrong hints.

Hypothesis C:

A conflict signature predicts when the model will resist or accept the wrong hint before the answer is emitted.

## Why Not Hallucination First

Hallucination matters more commercially, but the first mechanism card should avoid messy ground truth. Hallucination experiments need entity familiarity controls, retrieval controls, uncertainty calibration, and domain knowledge checks. Those are solvable but make a first card too easy to muddy.

Sycophancy lets the repo test the whole mechanism-card discipline with clean labels.

## Why Not Evaluation Awareness First

Evaluation awareness is strategically important, but it is extremely easy to confound with formatting or benchmark markers. A first card should not start with a target whose very definition is hard to separate from prompt artifacts.

## Initial Model And Tooling Choice

Current first stack:

- model family: Qwen3 for fast local iteration;
- working model: `Qwen/Qwen3-0.6B`;
- iteration reason: smallest Qwen3 target means the most controlled loops before escalation;
- comparison instruments: linear probes, activation directions, SAE features, attribution paths, and output-only baselines.

Artifact-rich comparison:

- `google/gemma-3-4b-it` remains useful if sparse-feature and transcoder artifacts become the bottleneck.

Escalation:

- try `Qwen/Qwen3-1.7B` or artifact-rich Gemma only after Qwen3-0.6B answers whether there is residual hidden control value beyond output logits.

The first card should not depend on a proprietary model.

Current implementation note:

- Qwen3-0.6B smoke passed the factual-ladder iteration gate.
- The controlled series found a real h14 intervention surface but no supported mechanism card.
- V7 showed that h13 and h14 carry nearly identical raw-direction option-logit effects, so the current surface is residual-stream transport rather than an h14-local mechanism.
- V8 tested a separately trained answer-prefix h14 signature and also failed to produce a clean dense mechanism.
- V9 found strong hint-source dependence: hint-line and hint-answer source masking sharply reduced wrong-hint agreement, but the carrying path is not yet localized.
- V10 tested single-head and single-layer attention-source localization; the best head recovered only 7/103 hard-bin truth-following.
- V11 tested cumulative all-head layer-band source masking; the best precise generation-query masks reached only 24/103 hard-bin truth for the full hint line and 22/103 for the hinted answer, far below V9's coarse input-mask result.
- V12 tested input-mask semantics and found that literal source rewrites nearly match V9-style input masking, while query-only source masks remain weak.
- V13 tested the remaining layout/tokenization caveat; character-matched and token-count-matched neutral rewrites reached only 52/103 hard-bin truth and did not rescue the mechanism claim.
- The broad Qwen3-0.6B MC-001 route is now diagnostic/control-only. Future mechanism-card work should escalate to a larger or artifact-rich stack.
- See [First Stack Decision](10_FIRST_STACK_DECISION.md) and [MC-001 Qwen3-0.6B Controlled V13 Layout-Parity Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V13_LAYOUT_PARITY_STATUS.md).

## First Mechanism Card Claim To Attempt

Narrow claim:

> In an open instruction-tuned language model, a hidden-state signature predicts whether the model will follow an incorrect user hint on objective questions, and an activation-level intervention changes that tendency without causing broad fluency or task degradation.

This claim is intentionally modest. If it fails, the failure is still useful.

## Kill Criteria

Retire the first target if:

- prompt-only anti-sycophancy instructions match or beat internal interventions;
- hidden signatures do not beat output-only confidence or logit baselines;
- interventions reduce sycophancy only by causing generic refusal;
- off-target performance drops enough to swamp the target gain;
- signatures fail on paraphrased prompts or unseen task families;
- wrong-hint behavior is too inconsistent to measure cleanly.

## Next Required Artifact

The first card preregistration lives at:

- [MC-001 sycophancy truth conflict](prereg/MC001_SYCOPHANCY_TRUTH_CONFLICT.md)
