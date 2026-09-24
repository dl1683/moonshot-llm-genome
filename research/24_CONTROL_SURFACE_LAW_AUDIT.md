# Control-Surface Law Audit

Date: 2026-07-01

Status: generated support audit implemented and validated.

This document interprets the machine-readable law-audit artifact:

> `data/control_surface_law_audit.json`

Builder:

> `code/control_surface_law_audit.py`

Commands:

```powershell
python code\control_surface_law_audit.py --write
python code\control_surface_law_audit.py
python code\validate_control_surface_atlas.py
```

## Purpose

The law hypotheses are not trophies. They are working predictions about where
small-LLM behaviors live and why mechanism claims pass, bound, or die.

The audit layer exists to stop those hypotheses from becoming another form of
storytelling. Every hypothesis must cite atlas rows and diagnostics that are
actually present in the current machine-readable atlas.

The audit checks:

- whether every cited evidence row exists;
- whether every cited diagnostic is observed in the cited evidence rows;
- whether any cited diagnostic is only vocabulary but not current evidence;
- whether every atlas row contributes to at least one law hypothesis;
- whether the status label is plausible for the amount of evidence cited;
- whether every hypothesis still has predictions, falsifiers, and next tests.

## Current Generated Facts

The current law audit covers:

- 9 hypotheses;
- 19 atlas rows;
- 60 observed diagnostics;
- 60 cited diagnostics;
- 19/19 rows with law support;
- 0 rows without law support;
- 0 unobserved cited diagnostics;
- 0 hypotheses with audit gaps.

Status counts:

- 1 `strong_doctrine`;
- 4 `supported_pattern`;
- 4 `tentative_pattern`.

Audit levels:

- 1 `doctrine_evidence_consistent`;
- 4 `pattern_evidence_consistent`;
- 4 `tentative_evidence_consistent`.

## Hypothesis Audit Table

| Hypothesis | Status | Audit Level | Evidence Rows | Cited Diagnostics |
| --- | --- | --- | ---: | ---: |
| `final_state_output_geometry_dominance` | `supported_pattern` | `pattern_evidence_consistent` | 5 | 16 |
| `lead_time_monitor_before_lever` | `supported_pattern` | `pattern_evidence_consistent` | 3 | 16 |
| `source_visible_lookup_localizes_more_than_parametric_override` | `supported_pattern` | `pattern_evidence_consistent` | 12 | 52 |
| `familiar_entities_can_collapse_to_lookup_keys` | `tentative_pattern` | `tentative_evidence_consistent` | 1 | 4 |
| `authority_pressure_creates_contrast_before_clean_substrate` | `tentative_pattern` | `tentative_evidence_consistent` | 10 | 34 |
| `coarse_source_ablation_overstates_circuit_locality` | `tentative_pattern` | `tentative_evidence_consistent` | 1 | 3 |
| `null_reliability_bottleneck` | `supported_pattern` | `pattern_evidence_consistent` | 3 | 8 |
| `behavior_substrate_first_or_everything_lies` | `strong_doctrine` | `doctrine_evidence_consistent` | 13 | 35 |
| `transfer_fails_at_reliability_before_primary_effect` | `tentative_pattern` | `tentative_evidence_consistent` | 1 | 3 |

## What Changed

The audit caught two evidence-boundary problems.

First, `behavior_substrate_first_or_everything_lies` cited
`REQUESTED_MODE_CONFUND`, which is a valid diagnostic vocabulary item but is not
currently observed in any atlas row. That citation was removed. The concept can
return when a live row actually exhibits it.

Second, `coarse_source_ablation_overstates_circuit_locality` was marked
`supported_pattern` even though its current evidence comes from one atlas row.
It was downgraded to `tentative_pattern`. The claim remains useful, but it is
not replicated enough to be treated as a supported cross-family pattern.

## Interpretation

This is the project becoming stricter about its own theory layer.

The atlas can now distinguish:

- a broad doctrine with many evidence rows;
- a supported pattern with cross-row support;
- a tentative pattern that is real enough to guide tests but not broad enough
  to anchor the genome.

That distinction matters. A project trying to map a "knowledge genome" can fail
by overclaiming laws as easily as it can fail by overclaiming circuits.

## What This Proves

It proves that the current law hypotheses are no longer unsupported prose.

They are now checked against:

- the atlas row ids;
- the atlas diagnostic set;
- the current cross-family comparison shape;
- the row-level mixture profiles;
- the current evidence-row coverage.

The validator now fails if the checked-in law audit is stale, if any law
hypothesis has evidence gaps, if a law cites an unobserved diagnostic, or if an
atlas row has no law support.

## What It Does Not Prove

It does not prove that the laws are true.

It does not prove transfer across new model families.

It does not prove the diagnostic ratios are stable under future bridge tasks.

It also does not replace experiments. The law audit only verifies that the
current hypotheses are evidence-bounded against the current atlas.

## Next Use

Every future behavior family should update three layers:

- the atlas row;
- the comparison artifact;
- the law audit.

The useful question after each future experiment is not just "did this pass?"

It is:

> Which law became stronger, weaker, newly falsified, or newly tentative?

That is the path from experiment pile to predictive control-surface science.
