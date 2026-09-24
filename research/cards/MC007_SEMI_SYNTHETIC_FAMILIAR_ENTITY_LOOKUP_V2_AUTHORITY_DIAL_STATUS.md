# MC007 Semi-Synthetic Familiar-Entity Lookup V2 Authority-Dial Status

Status: authority pressure created partial real-prior contrast, but the
behavior substrate failed parseability and holdout-balance gates.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP.md`
- V2 runner:
  `code/mc007_semi_synthetic_familiar_entity_lookup_v2_authority_dial.py`
- smoke result:
  `results/cards/MC007/mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_v2_authority_dial_20260701T011801.json`
- full result:
  `results/cards/MC007/mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_v2_authority_dial_20260701T012155.json`
- full result SHA256:
  `DA681D565EA805D75FA8ADC5832124C0954A96A92185C627392754812AD16AB2`

## Verdict

MC007 V2 did what V1 could not: it made real-world prior pressure visible
without printing the true capital in the prompt.

It still did not produce a valid hidden-state substrate.

The artifact diagnostic is:

```text
behavior_substrate_failed
```

The more precise atlas diagnosis is:

```text
authority-dial contrast with prompt-contract parseability failure
```

The failure is not empty. V2 establishes that the bridge from source-value
lookup toward factual override is controlled by prompt authority and question
framing, but the current contract is too parse-fragile and too holdout-weak to
support hidden-state probing.

## Structural Result

The full run was structurally clean:

- model: `Qwen/Qwen3-1.7B`;
- 40 country sources;
- 2 templates: `numeric_dial`, `natural_dial`;
- 7 panels per source/template;
- 560 generated rows;
- source-disjoint split;
- no duplicate record ids;
- no candidate collisions;
- no true-capital prompt leaks;
- selected prompt audit passed.

The selected behavior template was `numeric_dial`.

## Gate Result

| Criterion | Result |
| --- | --- |
| structural checks | pass |
| full source count is 40 | pass |
| synthetic control at least 90 percent artificial | pass: 39/40 |
| selected prompt audit passed | pass |
| primary binary rows at least 40 | pass: 149 |
| cross-dial contrast present | pass |
| within-panel contrast present | pass |
| primary parseability at least 90 percent | fail: 149/200 = 74.5 percent |
| cross-dial balance passed | fail |
| within-panel balance passed | fail |
| authority transition observed | fail under strict criterion |
| signature ready | false |

## Authority Curve

The important result is the authority curve. The table below reports generated
first-line labels for the 40 rows in each authority panel.

### Numeric Dial

| Authority | Artificial | Real Prior | Lure | Unparsed | Parseable | Artificial - Real Logit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 31 | 0 | 0 | 9 | 77.5 percent | +8.4672 |
| 70 | 26 | 0 | 0 | 14 | 65.0 percent | +8.4977 |
| 50 | 34 | 0 | 0 | 6 | 85.0 percent | +7.3453 |
| 30 | 13 | 13 | 2 | 12 | 70.0 percent | -0.5945 |
| 0 | 15 | 11 | 4 | 10 | 75.0 percent | -1.4695 |

The numeric dial gives the clearest transition. At authority 30 and 0, the
mean candidate margin flips against the artificial prompt value, and generated
answers include real capitals and lures. That is the first MC007 evidence that
familiar-entity prior pressure can beat prompt-local lookup under this bridge
family.

### Natural Dial

| Authority | Artificial | Real Prior | Lure | Unparsed | Parseable | Artificial - Real Logit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 24 | 0 | 0 | 16 | 60.0 percent | +5.0813 |
| 70 | 24 | 0 | 0 | 16 | 60.0 percent | +7.1500 |
| 50 | 25 | 0 | 1 | 14 | 65.0 percent | +4.6031 |
| 30 | 27 | 0 | 1 | 12 | 70.0 percent | +6.1547 |
| 0 | 26 | 3 | 2 | 9 | 77.5 percent | +3.9922 |

The natural wording stayed much closer to prompt-local table behavior. It also
had worse first-line parseability. This is not just a behavior contrast. It is
a prompt-format boundary.

## Primary Summary

For the selected `numeric_dial` template across all familiar authority panels:

- 200 primary rows;
- 119 artificial-value rows;
- 24 real-prior rows;
- 6 lure rows;
- 51 unparsed rows;
- parseability: 149/200 = 74.5 percent;
- mean artificial-minus-real first-token logit margin: +4.4492.

Non-holdout primary rows:

- 160 rows;
- 95 artificial-value rows;
- 21 real-prior rows;
- 6 lure rows;
- 38 unparsed rows;
- parseability: 122/160 = 76.25 percent.

Holdout primary rows:

- 40 rows;
- 24 artificial-value rows;
- 3 real-prior rows;
- 0 lure rows;
- 13 unparsed rows;
- parseability: 27/40 = 67.5 percent.

The holdout prior/lure count is too small and parseability is too weak.

## Best Within-Panel Near Miss

The best within-panel candidate was:

```text
numeric_dial / familiar_authority_0
```

All rows:

- 40 rows;
- 15 artificial-value rows;
- 11 real-prior rows;
- 4 lure rows;
- 10 unparsed rows;
- parseability: 30/40 = 75.0 percent;
- mean artificial-minus-real first-token logit margin: -1.4695.

Non-holdout:

- 11 artificial-value rows;
- 9 real-prior rows;
- 4 lure rows;
- 8 unparsed rows.

Holdout:

- 4 artificial-value rows;
- 2 real-prior rows;
- 0 lure rows;
- 2 unparsed rows.

This is close to a useful behavior contrast by label counts, but it fails the
parseability threshold. It is a V3 target, not a signature substrate.

## Null Result

The repaired answer-absent null contract was clean:

| Template | Unknown | Parseable | Unknown - Artificial Logit |
| --- | ---: | ---: | ---: |
| numeric_dial | 40/40 | 100.0 percent | +30.1411 |
| natural_dial | 40/40 | 100.0 percent | +30.1411 |

This matters because the first V2 smoke exposed a null-prompt regression. The
final V2 null contract repaired it without leaking the true capital.

## Interpretation

V1 showed:

> Familiar country names can collapse into table-lookup keys under a terse
> table-authoritative contract.

V2 adds:

> Explicit real-world-question pressure can partially overcome the table, but
> only under the numeric authority contract, and the resulting generated table
> is too parse-fragile for mechanism work.

The key axis is no longer "familiar names versus synthetic names." It is:

```text
prompt-local source authority
versus
real-world question authority
versus
first-line parse stability
```

This supports the larger atlas thesis: the bridge from lookup to knowledge is
not a single hidden surface. It is a mixture of prompt authority, output
geometry, source-value copying, and entity-specific prior strength.

## Allowed Claims

- MC007 V2 produced real-prior/lure outcomes without true-capital prompt leaks.
- Numeric authority pressure moved both generated labels and candidate margins.
- The selected V2 table is not parseable or balanced enough for hidden-state
  probing.
- Answer-absent null behavior can remain clean under the repaired null prompt.
- The best post-V2 repair target was the numeric authority-0 panel, but V3
  showed that target-only/compact notes and stricter answer-slot prompting did
  not solve the substrate.

## Forbidden Claims

- MC007 V2 is ready for hidden-state discovery.
- MC007 V2 found a knowledge-control mechanism.
- The authority dial proves causal control over real-world factual recall.
- The natural-language authority contract creates a clean prior-pressure
  substrate.
- The V2 contrast is independent of prompt text or output geometry.

## V3 Follow-Up

MC007 V3 has now been run:

`research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V3_PARSE_REPAIR_STATUS.md`

It tested target-only/compact task notes and stricter `Answer:` prompting. The
repair failed: the selected `compact_note` table reached only 55.0 percent
primary parseability and only 4/120 prior/lure primary rows.

MC007 should not proceed to hidden-state probing from V3. The next step is
either a materially different behavior-interface redesign or a diagnostic
closeout of the current MC007 route.

V4 has now been run:

`research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V4_AUTHORITY_INTERFACE_STATUS.md`

It tested explicit `CITY` and `SOURCE` fields. That redesign also failed:
selected-template synthetic and null controls broke, and source labels did not
stay coupled to city answers. The current MC007 route should be treated as a
diagnostic bridge unless a genuinely different behavior family is introduced.
