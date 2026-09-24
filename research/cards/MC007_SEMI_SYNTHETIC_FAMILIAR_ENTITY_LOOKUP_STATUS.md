# MC007 Semi-Synthetic Familiar-Entity Lookup Status

Status: source-value behavior substrate passed; contrast absent; not
signature-ready.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP.md`
- runner:
  `code/mc007_semi_synthetic_familiar_entity_lookup.py`
- smoke result:
  `results/cards/MC007/mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_behavior_20260701T010201.json`
- smoke SHA256:
  `417c8a5e3d09f3d1a542fc5543cd4b2733f76c9f53fdd6f2510dd7478edc477a`
- full result:
  `results/cards/MC007/mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_behavior_20260701T010554.json`
- full result SHA256:
  `0c83516b32088c5223e66c917f9202e6d97163290c458517a870fa4fba10ca17`

## Verdict

MC007 V1 passed a source-value behavior substrate under the selected
`lookup_only` template, but it did not produce a usable artificial-versus-real
contrast for hidden-state discovery.

The diagnostic class is:

```text
source_value_behavior_passed_contrast_absent
```

The behavior result is real and useful:

- 40 sources;
- 4 panels;
- 4 templates;
- 640 generated rows;
- source-disjoint split;
- no candidate collisions;
- prompt audit passed;
- selected template: `lookup_only`.

The selected template behaved like prompt-local lookup:

| Panel | Rows | Artificial | Real Prior | Lure | Unknown | Unparsed | Parseable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic control lookup | 40 | 39 | 0 | 0 | 0 | 1 | 39/40 = 97.5 percent |
| familiar entity lookup | 40 | 37 | 0 | 0 | 0 | 3 | 37/40 = 92.5 percent |
| familiar entity conflict | 40 | 38 | 0 | 0 | 0 | 2 | 38/40 = 95.0 percent |
| answer-absent null | 40 | 0 | 0 | 0 | 40 | 0 | 40/40 = 100.0 percent |

Primary familiar-entity rows across Panels B/C:

- 80 rows;
- 75 artificial-value rows;
- 0 real-prior rows;
- 0 lure rows;
- 5 unparsed rows;
- parseability: 75/80 = 93.75 percent.

Holdout primary rows:

- 16 rows;
- 14 artificial-value rows;
- 0 real-prior rows;
- 0 lure rows;
- 2 unparsed rows.

## Output Geometry

The next-token candidate margins already favor the artificial prompt-local
value on primary rows.

Selected-template mean first-token logit margins:

| Row Set | Artificial - Real Prior | Artificial - Lure | Unknown - Artificial |
| --- | ---: | ---: | ---: |
| primary familiar rows | +9.1902 | +11.6645 | -15.7906 |
| familiar entity lookup | +9.2781 | +11.8625 | -15.6273 |
| familiar entity conflict | +9.1023 | +11.4664 | -15.9539 |
| answer-absent null | -3.5975 | -2.2849 | +27.0057 |

This matters: MC007 V1 did not merely parse generated answers as artificial
values. The output interface itself strongly favored the artificial task value
over the real capital and lure candidates under the selected contract.

## Criteria

| Criterion | Result |
| --- | --- |
| smoke mode | pass: false |
| structural checks | pass |
| full source count is 40 | pass |
| primary binary rows at least 40 | pass: 75 |
| primary parseability at least 90 percent | pass: 93.75 percent |
| selected prompt audit passed | pass |
| Panel A artificial adherence at least 90 percent | pass: 97.5 percent |
| contrast present | fail: no real-prior or lure rows |
| contrast balance passed | fail |
| holdout source-disjoint | pass |

`passed` is true for behavior because the prompt-local source-value substrate
works. `signature_ready` is false because there is no artificial-versus-real
label contrast to probe.

## Interpretation

MC007 V1 gives the first bridge measurement between MC005 and MC006.

What it supports:

- familiar country names can behave as task-local lookup keys when the prompt
  contract is terse and table-authoritative;
- artificial prompt values dominate real-capital priors under the selected
  `lookup_only` template;
- answer-absent null behavior is clean at the behavior level: 40/40 null rows
  generated `UNKNOWN`;
- the bridge task currently sits closer to MC005-style source-value lookup than
  to MC006-style factual override.

What it blocks:

- no hidden-state discovery yet;
- no intervention yet;
- no claim that familiar entities expose a knowledge-control surface;
- no claim that MC007 has found a source-value circuit;
- no claim about transfer beyond Qwen3-1.7B.

The key boundary is not semantic-prior override. It is contrast absence.
Under this contract, the model almost never chooses the real capital or lure.

## Atlas Update

MC007 V1 adds a new atlas state:

```text
source_value_behavior_passed_contrast_absent
```

Canonical diagnostics:

- `CONTRAST_ABSENT`;
- `PROMPT_CONTRACT_PARSEABILITY`.

Lead-time state:

```text
not_reached
```

Intervention state:

```text
not_allowed
```

## Next Step

MC007 V2 should not run a hidden-signature search on this table.

The next repair should create real prior pressure without leaking the true
capital into the prompt. Useful directions:

- add a weaker-authority prompt where the table is described as a provisional
  note rather than the only authority;
- add an explicit choice between "task table" and "real-world memory" without
  naming the real capital;
- keep generated answers, strict first-line parsing, source-disjoint splits,
  and answer-absent nulls;
- preserve the `lookup_only` terse output contract as one arm, because it is
  the current clean source-value baseline.

Promotion to hidden-state work requires a repaired table with both artificial
and real-prior/lure outcomes on non-holdout and holdout splits.

V2 has now been run:

`research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V2_AUTHORITY_DIAL_STATUS.md`

It created partial real-prior/lure contrast under the numeric authority dial,
but parseability and holdout balance failed.

V3 has also now been run:

`research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V3_PARSE_REPAIR_STATUS.md`

It tested target-only/compact task notes and stricter answer-slot prompting.
That repair also failed: controls stayed clean, but the familiar-entity
conflict panel remained too unparseable and imbalanced for probing. MC007 is
still behavior-gated.

V4 has now been run:

`research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V4_AUTHORITY_INTERFACE_STATUS.md`

It tested explicit `CITY` and `SOURCE` fields. That route also failed:
selected-template controls broke, and source declarations did not reliably
match generated city answers. The current MC007 route is best treated as a
diagnostic bridge, not a probe substrate.
