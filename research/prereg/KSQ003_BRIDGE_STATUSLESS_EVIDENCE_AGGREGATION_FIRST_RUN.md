# KSQ003 Bridge Statusless Evidence Aggregation First Run

Status: behavior-substrate first-run preregistration; no hidden-state work.

Runner:

> `code/ksq003_bridge_statusless_evidence_aggregation_first_run.py`

Default result:

> `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_first_run.json`

Default 10-source smoke result:

> `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json`

Status card:

> `research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md`

## Purpose

Test a statusless evidence-aggregation bridge before any hidden-state
signature work. Branch selection depends on symbol/parity evidence
about the queried element, not source status labels, row codes,
checksums, cross-table consistency, or row-local atomic-number claims.

## Panels

- `source_local_direct_control`
- `learned_fact_direct_control`
- `all_evidence_fit_conflict`
- `one_evidence_mismatch_conflict`
- `symbol_only_ablation`
- `parity_only_ablation`
- `answer_absent_and_side_null`

## Decision Boundary

Promote only to behavior-substrate admission if structural checks,
direct controls, conflict mixture, ablations, nulls, source-disjoint
holdout, and candidate/output baselines pass together.

Death rule: kill the candidate if the learned branch collapses under
table pressure, if evidence features behave like visible status
labels, if nulls fail, or if output/candidate baselines explain the
branch.

Forbidden claims:

- KSQ003 is a mechanism card.
- KSQ003 licenses hidden-state search before the behavior gate passes.
- KSQ003 proves a learned-memory bridge.
