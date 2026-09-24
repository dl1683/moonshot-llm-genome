# KSQ012 Function-Assignment Wrapper Repair

Status: KSQ011 diagnostic follow-up.

Runner:

> `code/ksq012_function_assignment_wrapper_repair.py`

Artifacts:

> `results/cards/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR/ksq012_function_assignment_wrapper_repair_first_run.json`

> `results/cards/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR/ksq012_function_assignment_wrapper_repair_smoke_limit10.json`

> `results/cards/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR/ksq012_function_assignment_wrapper_repair_full_behavior.json`

## Claim Under Test

KSQ011 showed that function-like assignment strings form an answer
channel broader than exact answer_for spelling. This packet keeps the
selected tag_rows codebook bridge fixed and tests whether wrappers,
comment marks, fenced text, cut markers, masked calls, split function/
value rows, or unrelated-entity calls suppress that channel.

## Promotion Rule

Promote only to behavior-substrate candidate status if exact bridge
answers, raw answer_for remains an active positive control, all wrapper
repair panels answer the counted bridge rather than the alternate,
assignment-only wrapper controls abstain, query-only abstains, selected
prompt audit passes, source-disjoint holdout passes, and margins are
reported.

## Kill / Boundary Rule

If raw answer_for no longer reproduces, the packet is not testing the
KSQ011 pressure. If wrappers still override the bridge, export wrapper
failure. If wrapper-only controls reproduce the value, export wrapper
control leakage. If only split or unrelated-entity forms pass, export a
partial boundary rather than a repair.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden unless a later full behavior
  run and margin report admit only a signature screen.
