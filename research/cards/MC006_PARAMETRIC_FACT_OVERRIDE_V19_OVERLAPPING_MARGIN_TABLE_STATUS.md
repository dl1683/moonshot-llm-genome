# MC006 Parametric Fact Override V19 Overlapping-Margin Table Status

Status: expanded prompt bank improved behavior balance and margin proximity, but
strict candidate/final-output margin overlap still failed.

Date: 2026-07-01

## Artifact

- design source:
  `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`
- runner:
  `code/mc006_parametric_fact_override_v19_overlapping_margin_table.py`
- source V14 behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- source V18 margin diagnostic artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v18_margin_matched_leadtime_20260701T022343.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json`
- result SHA256:
  `aea4913e580da82f52f301af9360749b8b048d95bcfcb20db59c7411dd5da29f`

## Verdict

V19 does not make MC006 signature-ready.

The diagnostic class is:

```text
non_holdout_candidate_margin_overlap_failed
```

The useful result is a near miss. V19 built a 400-row generated-answer prompt
bank over 40 sources and 10 real-after-fiction prompt contracts. It found a
template that repaired label balance better than V14/V18, but the binary true
and override rows still remained separated by candidate-score and final-output
margins.

## What V19 Tested

V19 asked whether the V18 blocker was specific to the V14 `fake_mapping_warning`
table. It expanded the prompt bank to ten variants that changed how strongly
the prompt marked the fictional mapping as separate from real-world geography:

- `fake_mapping_warning`;
- `fictional_code_ignore`;
- `untrusted_note_real`;
- `possibly_mistaken_note`;
- `separate_task_weak`;
- `reference_line_audit`;
- `memory_interference`;
- `ambiguous_note`;
- `game_code_then_geo`;
- `minimal_conflict`.

Each row used greedy generated answers, strict NFKD first-line parsing, no
true-capital prompt leaks, candidate mean-logprob margins, and final next-token
true-minus-override margins.

The selection rule used non-holdout evidence:

1. maximize the smaller of non-holdout true and override counts;
2. maximize the smaller of non-holdout candidate/final matched-pair counts at
   z <= 0.5;
3. maximize non-holdout candidate/final overlap flags;
4. maximize non-holdout binary rows;
5. minimize side rows.

## Selected Template

The selected template was:

```text
separate_task_weak
```

Selected-template behavior:

| Measure | Value |
| --- | ---: |
| total rows | 40 |
| binary rows | 35 |
| side rows | 5 |
| non-holdout true | 11 |
| non-holdout override | 16 |
| holdout true | 4 |
| holdout override | 4 |
| holdout side rows | 0 |
| prompt leaks | 0 |

This is a real behavior-table improvement over V18 for the margin-matching
goal: holdout has both labels at 4/4 rather than V18's 6/2 split, and the
selected prompt creates weaker global-margin separation.

## Margin Result

The selected template still failed strict overlap.

Holdout candidate-score margin, z-scored from non-holdout rows:

- true rows: 4;
- override rows: 4;
- true z-range: 0.2882 to 0.7235;
- override z-range: -0.5745 to 0.0311;
- overlap exists: false;
- separation gap: 0.2571;
- matched pairs at z <= 0.5: 4.

Holdout final next-token margin, z-scored from non-holdout rows:

- true rows: 4;
- override rows: 4;
- true z-range: 0.1146 to 0.8253;
- override z-range: -0.5752 to -0.1363;
- overlap exists: false;
- separation gap: 0.2508;
- matched pairs at z <= 0.5: 4.

The pooled prompt bank was also only a near miss. Across all templates, holdout
binary rows were 42 true and 15 override. Candidate-score holdout separation
gap shrank to 0.0552z, and final-output holdout separation gap shrank to
0.0612z, but final-output class ranges still did not overlap. Pooling would
also introduce a prompt-template confound unless handled as a separate
matched-template design.

## Criteria

| Criterion | Result |
| --- | --- |
| structural checks pass | pass |
| selected binary rows at least 30 | pass: 35 |
| selected non-holdout true at least 6 | pass: 11 |
| selected non-holdout override at least 6 | pass: 16 |
| selected non-holdout candidate margin overlap | fail |
| selected non-holdout final margin overlap | fail |
| selected holdout true at least 2 | pass: 4 |
| selected holdout override at least 2 | pass: 4 |
| selected holdout candidate margin overlap | fail |
| selected holdout final margin overlap | fail |
| selected holdout candidate pairs at z <= 0.5 | pass: 4 |
| selected holdout final pairs at z <= 0.5 | pass: 4 |
| no true-answer prompt leak | pass |

## Interpretation

V19 improves the MC006 map:

- V18 showed that the V14 table had no holdout matching region.
- V19 shows that weaker prompt authority can create balanced true/override
  generated labels and near-matched margin pairs.
- But even in the best selected template, the generated label is still
  separated by candidate-score and final-output geometry.

Allowed claim:

> MC006 can build a better balanced generated-answer table with near-matched
> global-margin pairs, but the current expanded prompt bank still fails strict
> overlapping-margin promotion.

Forbidden claims:

- V19 supports hidden-state probing;
- V19 supports intervention;
- `separate_task_weak` is an overlapping-margin table;
- pooling templates solves MC006 without a prompt-format control;
- near-pair matching at z <= 0.5 is the same as strict margin overlap.

## Next Step

The next MC006 branch has two defensible options:

1. build a matched-template or per-source selection design that treats prompt
   template as a first-class confound and targets strict final-margin overlap;
2. run a deliberately bounded pair-matched lead-time diagnostic on V19 rows,
   explicitly labeled as approximate matching rather than mechanism promotion.

Until then, MC006 remains:

- behavior-supported by V14;
- lead-time-monitor-supported by V16;
- additive-steering-failed by V17;
- same-table margin-matching-blocked by V18;
- overlapping-margin table search near-missed by V19;
- not intervention-ready.
