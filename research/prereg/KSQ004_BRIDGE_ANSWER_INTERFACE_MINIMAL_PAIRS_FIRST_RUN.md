# KSQ004 Bridge Answer-Interface Minimal Pairs First Run

Status: behavior-substrate first-run preregistration; no hidden-state work.

Runner:

> `code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py`

Default result:

> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_first_run.json`

Default 10-source smoke result:

> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json`

Full behavior result:

> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json`

Status card:

> `research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md`

## Purpose

Test whether bridge-style local-vs-learned behavior survives a shared
bare numeric answer interface. The run is designed to kill answer-shape
and branch-shortcut explanations before hidden-state search.

## Panels

- `matched_minimal_pairs`
- `local_learned_direct_controls`
- `minimal_pair_conflict`
- `side_answer_leakage`
- `null_and_holdout`

## Decision Boundary

Promote only to behavior-substrate admission if matched minimal
pairs parse under the same answer schema, direct local and learned
controls both pass, conflict minimal pairs produce the requested
branch on non-holdout and holdout rows, side-answer leakage stays
low, answer-absent rows abstain, and candidate/output baseline
reporting is present.

Death rule: export `ANSWER_INTERFACE_BRANCH_SHORTCUT` if side
answers or answer-shape shortcuts explain the behavior; export
`BRIDGE_MINIMAL_PAIR_CONTRAST_ABSENT` if balancing the answer
interface removes the local-vs-learned contrast.

Forbidden claims:

- KSQ004 is a mechanism card.
- KSQ004 licenses hidden-state search before the behavior gate passes.
- KSQ004 proves a learned-memory bridge or answer-interface control surface.
