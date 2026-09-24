# Control-Surface Comparison Snapshot

Date: 2026-07-01

Status: generated comparison layer implemented and validated.

This document interprets the machine-readable comparison artifact:

> `data/control_surface_comparison.json`

The law audit that checks whether the theory layer is actually supported by the
atlas is:

> `data/control_surface_law_audit.json`

The next-experiment queue that turns the law layer into prioritized tests is:

> `data/control_surface_next_experiment_queue.json`

Builder:

> `code/control_surface_comparison.py`

Commands:

```powershell
python code\control_surface_comparison.py --write
python code\control_surface_comparison.py
python code\control_surface_law_audit.py --write
python code\control_surface_next_queue.py --write
python code\validate_control_surface_atlas.py
```

## Purpose

The reviewer comments make one point sharper than the old framing:

> The genome is not the pile of promoted mechanism cards. The genome is the
> measured distribution of where each behavior lives and why each mechanism
> claim passes, bounds, or dies.

The comparison layer makes that distribution explicit. It reads the atlas and
the artifact index, then aggregates verdicts, lead-time states, intervention
states, mixture axes, row diagnostics, artifact diagnostics, null-boundary
signals, artifact coverage, and row-claim consistency.

This is the first project-level answer to:

> What is the current shape of the small-model control-surface map?

## Current Generated Facts

The current comparison has 19 atlas rows.

Verdicts:

- 0 promoted mechanism cards;
- 1 bounded mechanism card;
- 1 failed mechanism-card route;
- 17 diagnostic notes.

Lead-time states:

- 2 `lead_time_monitor_only`;
- 3 `lead_time_output_shadow`;
- 1 `zero_lead_time`;
- 1 `not_primary_axis`;
- 12 `not_reached`.

Intervention states:

- 1 `causal_dirty`;
- 2 `behavior_control_only`;
- 2 `failed`;
- 12 `not_allowed`;
- 2 `not_tested`.

Genome-shape ratios:

- promoted mechanism ratio: 0.000000;
- bounded mechanism ratio: 0.052632;
- diagnostic-or-failed ratio: 0.947368;
- lead-time monitor-or-shadow ratio: 0.263158;
- intervention not-allowed-or-failed ratio: 0.736842;
- output-margin-confounded row ratio: 0.315789;
- behavior-substrate-failed row ratio: 0.368421.

Mixture-axis counts:

| Axis | Counts | Strong/Bounded/Failed Ratio |
| --- | --- | --- |
| Prompt authority | dominant 9; high 10 | 1.000000 |
| Prompt format | dominant 1; high 12; medium 6 | 0.684211 |
| Source-token dependence | dominant 1; high 11; medium 5; untested 2 | 0.631579 |
| Output geometry | dominant 2; high 12; medium 1; untested 4 | 0.736842 |
| Lead-time internal signal | high 1; medium 1; low 4; none 13 | 0.052632 |
| Local internal path | high 1; low 5; none 3; untested 10 | 0.052632 |
| Causal control | bounded 1; behavior_only 2; failed 2; untested 14 | 0.263158 |
| Null locality | bounded 5; behavior_only 9; failed 1; untested 4 | 0.789474 |
| Transfer | failed 1; medium 2; low 2; untested 14 | 0.052632 |

Artifact coverage:

- 53 parsed artifacts;
- 46 summary-schema artifacts;
- 7 legacy-schema artifacts;
- 53 artifacts with extracted metrics;
- all 19 atlas rows have parsed result-artifact support;
- no atlas rows remain prose-only.

Claim-audit coverage:

- 19 rows with linked result artifacts;
- 19 rows with extracted metrics;
- 19 rows with family-level claim checks;
- 43 enabled family-level closure checks;
- 0 rows without artifacts;
- 0 rows without metrics;
- 0 rows without family checks;
- 19 rows at `family_checked_with_metrics`;
- 0 rows at `family_checked_partial_metrics`;
- 14 rows with failed or false criteria exposed in linked artifacts.

The closure checks now include three MC005 boundaries, eight MC006 boundaries,
and granular bridge-route checks for MC007-MC016. MC005 is checked for
layers-24-26 attention-write mediation, nonzero null flips, and the non-simple
margin-cutoff null boundary. MC006 is checked for the V16 output/candidate
confound, V17 additive-steering failure, V21 pair-matching margin-baseline
failure, V22 source/path shadow, V24 delayed-city monitor-only status, V25
candidate-decoupled shuffle overfit, V28 one-transfer-template-not-bank
outcome, and the combined shuffle/transfer failure. The bridge checks split
MC007 into V1 source-lookup/no-conflict, V2 authority-dial parseability
failure, V3 parseability-repair failure, and V4 source-declaration control
failure; MC008 into direct-controls-before-null and null-repaired/conflict-
absent boundaries; MC009 into membership-control and typed-slot-control
tradeoffs; MC010 into two-hop direct-control and table-dominance failure;
MC011 into clean numeric controls plus complete numeric conflict collapse;
MC012 into reliability-labeled behavior contrast plus prompt-channel blockage;
MC013 into statused positive-control reproduction plus matched-ablation
contrast collapse; MC014 into direct-control cleanliness plus
calibration-inference conflict collapse; and MC015 into direct-control
cleanliness plus mixed-output parity-gate rule-following failure; and MC016
into direct-control cleanliness plus alphabet-gate local collapse.

Claim-consistency coverage:

- 120 generic row-claim conditions checked;
- 0 claim-consistency contradictions;
- 0 rows with claim-consistency contradictions.

Law-audit coverage:

- 9 law hypotheses;
- 1 `strong_doctrine`, 4 `supported_pattern`, and 4 `tentative_pattern`;
- 9/9 hypotheses at evidence-consistent audit levels;
- 19/19 atlas rows with law support;
- 0 unobserved diagnostics cited as law evidence.

Next-queue coverage:

- 19 queue items;
- 5 `immediate` items;
- 4 `high` items;
- 9 `medium` items;
- 1 `watch` item;
- top pressure: build a bridge where the answer rule is independent of visible
  status text and prove rule-following, full bridge-substrate gate, transfer/null panels, transfer
  promotion rule, and source-ablation locality controls.

Null-boundary summary:

- 18 artifacts expose null criteria;
- 6 artifacts contain failed null criteria;
- failed null classes include answer-absent unknown failures, answer-absent
  parseability failures, hidden/shuffle-null failures, and transfer hidden-null
  failures.

## Interpretation

The current map is not mechanism-card-rich. It is diagnostic-rich.

That is not a weak result. It is the first quantified shape of the project:

- prompt authority is present everywhere;
- output/candidate geometry is often a better explanation than hidden probes;
- clean local internal paths are rare;
- causal intervention is more often blocked, dirty, or behavior-only than clean;
- null locality is a central boundary, not a cleanup detail;
- lead-time monitors exist, but they usually do not become levers;
- transfer is thin and mostly untested.

This supports the revised objective:

> Build the machine that makes false mechanism claims die quickly, and preserve
> the typed deaths as the control-surface atlas.

## What This Proves

It proves that the project now has a comparable cross-family object. MC001-MC016
are no longer only stories in prose; they are rows with shared axes and a
generated comparison. The generated comparison now also has a claim-audit
section, and atlas validation fails if any row has no result artifact, no
extracted metrics, no family-level claim check, or only partial metric
coverage.

It also has a claim-consistency section. Atlas validation now fails if a row's
high-level verdict, intervention state, lead-time state, null-locality field,
behavior-gate diagnostic, output-confound diagnostic, or signature-causality
diagnostic contradicts the normalized artifact evidence.

It also proves that the largest near-term insight is not "we found the truth
vector." The insight is:

> Small-model behavior is distributed across prompt authority, source-token
> dependence, output geometry, lead-time monitors, local internal paths, null
> boundaries, and transfer fragility. The distribution itself is the genome
> object.

## What It Does Not Prove

It does not prove the full knowledge genome.

It does not prove that all future behavior families will have the same ratios.

It does not prove that current diagnostic ratios are stable across model
families, model sizes, or better bridge tasks.

It also does not replace raw artifacts. The comparison is an index and
measurement layer; status cards and result JSON remain the evidence.

## Next Use

Every new branch should update the comparison layer.

The next useful comparison improvements are:

- deepen family-specific validators so row verdicts, null-locality claims,
  output/candidate controls, and intervention claims become more automatically
  auditable;
- turn more allowed/forbidden claims into explicit generated consistency rules;
- keep the law audit current so each future row either strengthens, weakens,
  or falsifies a named law hypothesis;
- keep the next-experiment queue current so branch choice is tied to law
  pressure rather than narrative momentum;
- add trend fields after future MC012-MC016-style bridge attempts, so the atlas can
  say whether a new behavior family changes the mixture ratios or only adds
  another typed failure.

The immediate research target is no longer a prettier single card. It is a
predictive comparison table: given a behavior and prompt contract, forecast
which surface will dominate, which baseline will kill the claim, and what kind
of intervention boundary will appear.
