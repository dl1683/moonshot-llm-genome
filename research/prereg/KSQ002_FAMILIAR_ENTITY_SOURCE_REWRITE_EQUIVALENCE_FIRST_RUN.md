# KSQ002 Familiar Entity Source-Rewrite Equivalence First Run

Status: behavior-substrate first-run preregistration; no hidden-state work.

Runner:

> `code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py`

Default result:

> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_first_run.json`

Default 10-source smoke result:

> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json`

Full behavior result:

> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json`

Status card:

> `research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md`

## Purpose

Test whether familiar-entity artificial source lookup survives neutral
rewrites and disappears when the source row is deleted or when only the
query country remains.

## Panels

- `baseline_source_value_lookup`
- `neutral_rewrite_lookup`
- `source_deletion`
- `query_only_control`
- `source_disjoint_rewrite_holdout`
- `rewrite_output_geometry_audit`

## Decision Boundary

Promote only to behavior-substrate admission if baseline source-value
lookup passes, neutral rewrites preserve it within the predeclared
delta, source deletion and query-only controls do not reproduce the
source value, source-disjoint rewrite holdout passes, and
candidate/output baseline reporting is present.

Death rule: kill if neutral rewriting breaks the behavior, if source
deletion preserves the source value, if query-only text reproduces
the source value, or if output/candidate geometry explains the split.

Forbidden claims:

- KSQ002 is a mechanism card.
- KSQ002 licenses hidden-state search before the behavior gate passes.
- KSQ002 proves a source-channel or knowledge-control surface.
