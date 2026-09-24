# KSQ011 Answer_For Syntax Ablation

Status: KSQ010 diagnostic follow-up.

Runner:

> `code/ksq011_answer_for_syntax_ablation.py`

Artifacts:

> `results/cards/KSQ011_ANSWER_FOR_SYNTAX_ABLATION/ksq011_answer_for_syntax_ablation_first_run.json`

> `results/cards/KSQ011_ANSWER_FOR_SYNTAX_ABLATION/ksq011_answer_for_syntax_ablation_smoke_limit10.json`

> `results/cards/KSQ011_ANSWER_FOR_SYNTAX_ABLATION/ksq011_answer_for_syntax_ablation_full_behavior.json`

## Claim Under Test

KSQ010 failed because answer_for alternates overrode two-stage codebook
bridges. This packet holds the selected tag_rows codebook interface
fixed and varies only the adversarial alternate syntax.

## Promotion Rule

Promote only to behavior-substrate candidate status if the exact bridge
answers, every adversarial alternate variant fails to override it,
adversary-only and query-only controls abstain, selected prompt audit
passes, source-disjoint holdout passes, and margins are reported.

## Kill / Boundary Rule

If exact bridge lookup fails, export a bridge-positive failure. If only
exact answer_for overrides, export lexical dominance. If multiple
function-like variants override, export function-assignment dominance.
If nonfunction variants override, export broad value-salience dominance.
Treat the typed failure as the datum.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden unless a later full behavior
  run and margin report admit only a signature screen.
