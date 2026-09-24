# MC006 Parametric Fact Override V20 Strict-Overlap Selection Audit Status

Status: strict final-output margin overlap is absent from the V19 binary row
bank; only bounded pair-matched diagnostics remain available from V19.

Date: 2026-07-01

## Artifact

- design source:
  `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`
- runner:
  `code/mc006_parametric_fact_override_v20_strict_overlap_selection_audit.py`
- source V19 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json`
- source V19 SHA256:
  `aea4913e580da82f52f301af9360749b8b048d95bcfcb20db59c7411dd5da29f`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit_20260701T025043.json`
- result SHA256:
  `d6f677aab7fe38f681124fc939cd18d8a322c35d694d81750fcf6f76c2598708`

## Verdict

V20 does not make MC006 signature-ready.

The diagnostic class is:

```text
strict_final_margin_overlap_absent
```

V20 is an offline upper-bound audit. It asks whether the V19 400-row prompt
bank contains any strict-overlap behavior substrate that could be rescued by
matched-template or per-source selection.

It does not. The entire V19 binary row bank remains separated by final
next-token output margin.

## Source Check

V20 validated the V19 artifact:

- expected run type: pass;
- expected V19 diagnostic: `non_holdout_candidate_margin_overlap_failed`;
- source not signature-ready: pass;
- 400 generated records: pass;
- template summary present: pass.

The V19 bank contains:

- 400 generated rows;
- 257 binary true/override rows;
- 10 prompt templates;
- strict NFKD parsing from V19;
- candidate-score margins from V19;
- final next-token margins from V19.

## Pooled Bank Result

V20 pooled all binary rows across templates as an upper bound. This is not a
promotion substrate because prompt template is a confound, but it answers
whether any strict final-margin overlap exists in the bank at all.

Pooled holdout binary rows:

- true rows: 42;
- override rows: 15.

Pooled holdout candidate-score margin:

- overlap exists: false;
- z separation gap: 0.0552.

Pooled holdout final next-token margin:

- true z-range: -0.4833 to 1.6183;
- override z-range: -1.5239 to -0.5445;
- overlap exists: false;
- z separation gap: 0.0612.

This means every holdout override row still has lower final-output margin than
the lowest holdout true row in the pooled binary bank. No subset can create
strict final-margin overlap unless new generated rows are created.

## Template Result

No single template passed the full strict-overlap gate.

V20 criteria:

| Criterion | Result |
| --- | --- |
| source artifact valid | pass |
| pooled non-holdout candidate overlap | pass |
| pooled non-holdout final overlap | fail |
| pooled holdout candidate overlap | fail |
| pooled holdout final overlap | fail |
| any single-template full gate passed | fail |
| any template non-holdout final overlap | fail |
| any template holdout final overlap | fail |
| pooled holdout joint pairs at z <= 0.5 | pass |
| selected-template holdout joint pairs at z <= 0.5 | pass |

## Pair-Matched Boundary

V20 found near-pair evidence:

- pooled holdout joint candidate/final pairs at z <= 0.5: 28;
- selected-template holdout joint candidate/final pairs at z <= 0.5: 3.

This supports a bounded next diagnostic:

> A V19-row pair-matched lead-time audit is allowed only if labeled
> approximate and diagnostic. It cannot be promoted as a mechanism-grade
> signature because strict final-output overlap is absent.

## Interpretation

V20 closes the current strict-overlap route inside the V19 prompt bank:

- V18: V14 has no same-table matching region.
- V19: broader prompt pressure creates balanced labels and near-matched pairs.
- V20: even the pooled V19 binary bank lacks strict final-output overlap, so
  matched-template/per-source selection cannot rescue strict promotion without
  new rows.

Allowed claim:

> The V19 prompt bank contains approximate pair-matched examples for a bounded
> lead-time diagnostic, but no strict final-output-overlap table for
> mechanism-grade hidden-state work.

Forbidden claims:

- V20 supports hidden-state probing for mechanism promotion;
- V20 supports intervention;
- matched-template or per-source selection can rescue strict overlap inside
  the existing V19 bank;
- pooled prompt-bank overlap is a valid prompt-controlled substrate;
- approximate pair matching is the same as strict output/candidate overlap.

## Next Step

The next MC006 branch should choose explicitly:

1. **Bounded diagnostic:** run a pair-matched lead-time audit on V19 rows,
   labeled as approximate and non-promotional.
2. **New behavior construction:** generate new rows with targeted prompt
   pressure near the final-margin boundary.
3. **Source/path route:** stop pursuing scalar table separation and audit
   source-token or fictional-value paths directly.

Until then, MC006 remains:

- behavior-supported by V14;
- lead-time-monitor-supported by V16;
- additive-steering-failed by V17;
- same-table margin-matching-blocked by V18;
- overlapping-margin prompt-bank near-missed by V19;
- strict-overlap selection-blocked by V20;
- not intervention-ready.
