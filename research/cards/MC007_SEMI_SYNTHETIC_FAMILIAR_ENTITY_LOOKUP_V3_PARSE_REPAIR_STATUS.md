# MC007 Semi-Synthetic Familiar-Entity Lookup V3 Parse-Repair Status

Status: parseability repair failed; answer-slot strictness did not create a
hidden-state-ready behavior substrate.

Date: 2026-07-01

## Artifact

- V3 runner:
  `code/mc007_semi_synthetic_familiar_entity_lookup_v3_parse_repair.py`
- target-only smoke result:
  `results/cards/MC007/mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_v3_parse_repair_20260701T013214.json`
- full result:
  `results/cards/MC007/mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_v3_parse_repair_20260701T013926.json`
- full result SHA256:
  `6AEE0A1E62DF545B9421005289BFFCA03B29EFD3FC2E517B9DF8DE486F4A9541`

## Verdict

MC007 V3 tested the most direct repair implied by V2: reduce the table width,
use target-only or compact task notes, enforce a stricter `Answer:` slot, and
preserve the real-world capital question without printing the true capital.

The artifact diagnostic is:

```text
parseability_repair_failed
```

The result is not signature-ready:

```text
passed = false
signature_ready = false
```

V3 did improve the clean side controls, but it did not produce a parseable,
balanced artificial-versus-real behavior table for familiar countries.

The important conclusion is:

> Answer-slot strictness is not enough. In MC007, the task-note prompt itself
> either pulls the model back toward artificial prompt-local lookup or produces
> explanatory first lines. The bridge substrate is still a prompt-contract
> problem, not a hidden-state problem.

## Structural Result

The full run was structurally clean:

- model: `Qwen/Qwen3-1.7B`;
- 40 country sources;
- 2 templates: `target_only_note`, `compact_note`;
- 6 panels per source/template;
- 480 generated rows;
- source-disjoint split;
- no duplicate record ids;
- no candidate collisions;
- no true-capital prompt leaks;
- selected prompt audit passed.

The selected behavior template was `compact_note`.

## Gate Result

| Criterion | Result |
| --- | --- |
| structural checks | pass |
| full source count is 40 | pass |
| synthetic control at least 90 percent artificial | pass: 37/40 |
| selected prompt audit passed | pass |
| primary binary rows at least 40 | pass: 66 |
| cross-panel contrast present | pass |
| within-panel contrast present | pass |
| primary parseability at least 90 percent | fail: 66/120 = 55.0 percent |
| cross-panel behavior gate | fail |
| within-panel behavior gate | fail |
| signature ready | false |

V3 therefore remains behavior-gated. It should not be used for hidden-state
signature search or intervention.

## Template Result

### Target-Only Note

Primary target-note rows across authority 50/30/0:

- 120 rows;
- 36 artificial-value rows;
- 0 real-prior rows;
- 0 lure rows;
- 84 unparsed rows;
- parseability: 36/120 = 30.0 percent;
- mean artificial-minus-real first-token logit margin: +4.8250.

Target-only notes did not preserve real-prior pressure. They mostly produced
either artificial task-city answers or unparseable first lines.

### Compact Note

Primary target-note rows across authority 50/30/0:

- 120 rows;
- 62 artificial-value rows;
- 2 real-prior rows;
- 2 lure rows;
- 54 unparsed rows;
- parseability: 66/120 = 55.0 percent;
- mean artificial-minus-real first-token logit margin: +6.0828.

The compact note preserved a tiny amount of prior/lure behavior, but not enough
for a balanced behavior substrate. The source-disjoint holdout had only 1
prior/lure row across 24 primary rows.

## Best Within-Panel Near Miss

The selected within-panel candidate was:

```text
compact_note / target_note_authority_0
```

All rows:

- 40 rows;
- 19 artificial-value rows;
- 2 real-prior rows;
- 2 lure rows;
- 17 unparsed rows;
- parseability: 23/40 = 57.5 percent;
- mean artificial-minus-real first-token logit margin: +2.7313.

Non-holdout:

- 13 artificial-value rows;
- 1 real-prior row;
- 2 lure rows;
- 16 unparsed rows;
- parseability: 16/32 = 50.0 percent.

Holdout:

- 6 artificial-value rows;
- 1 real-prior row;
- 0 lure rows;
- 1 unparsed row;
- parseability: 7/8 = 87.5 percent.

This is worse than the V2 numeric authority-0 near miss for contrast. V2's
numeric authority-0 panel had 15 artificial and 15 real-prior/lure rows at 75.0
percent parseability. V3 reduced table-width/answer-slot noise but largely
collapsed the useful contrast.

## Control Panels

The control panels show why this is a real prompt-contract failure rather than
a broken parser everywhere.

| Panel | Template | Artificial | Real Prior | Lure | Unknown | Unparsed | Parseable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic lookup | target-only | 37 | 0 | 0 | 0 | 3 | 92.5 percent |
| synthetic lookup | compact | 37 | 0 | 0 | 0 | 3 | 92.5 percent |
| real-world control | target-only | 0 | 34 | 4 | 0 | 2 | 95.0 percent |
| real-world control | compact | 0 | 34 | 4 | 0 | 2 | 95.0 percent |
| answer-absent null | target-only | 0 | 0 | 0 | 40 | 0 | 100.0 percent |
| answer-absent null | compact | 0 | 0 | 0 | 40 | 0 | 100.0 percent |

Direct geography is parseable. Synthetic lookup is mostly parseable. Nulls are
perfectly parseable. The failure lives specifically in the target-note conflict
prompt.

## Interpretation

V1 taught:

> Familiar countries can collapse into prompt-local lookup keys under a terse
> table-authoritative contract.

V2 taught:

> Explicit authority pressure can create real-prior/lure contrast, but the
> generated-answer table becomes parse-fragile and holdout-weak.

V3 teaches:

> Simple answer-slot strictness and target-only task notes do not repair that
> fragility. They clean up side controls but either preserve artificial
> prompt-local dominance or induce unparseable explanatory first lines in the
> conflict panel.

This is a useful negative result. It narrows the bridge problem: the next MC007
attempt should not be another small answer-format tweak. It needs a different
behavior contract that preserves contrast and parseability at the same time.

## Allowed Claims

- MC007 V3 was structurally clean and preserved source-disjoint holdouts.
- V3 repaired or preserved clean direct-control behavior: real-world control
  was 38/40 prior/lure and answer-absent nulls were 40/40 `UNKNOWN`.
- V3 did not produce a parseable, balanced familiar-entity conflict substrate.
- The strict answer slot did not rescue the V2 authority-dial route.
- MC007 remains a behavior-substrate research line, not a hidden-state
  mechanism line.

## Forbidden Claims

- MC007 V3 is ready for hidden-state probing.
- MC007 V3 found a familiar-entity knowledge-control signature.
- Stricter answer-slot prompting solves MC007 parseability.
- The small number of real-prior/lure outputs is enough to justify a probe.
- V3 proves that familiar-country prior pressure is absent. It only proves that
  this prompt contract failed to measure it cleanly.

## Next Step

Declare the V3 repair route failed.

V4 has now been run:

`research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V4_AUTHORITY_INTERFACE_STATUS.md`

It tested explicit `CITY` and `SOURCE` fields. That redesign also failed:
selected-template controls broke, and source declarations did not reliably
match generated city answers.

The current MC007 route should be treated as a diagnostic bridge unless a
genuinely different behavior family is introduced. A future branch should not
be another prompt-only repair:

- keep V1 as the clean source-value baseline;
- keep V2 numeric authority-0 as the best contrast baseline;
- stop relying on answer-slot strictness alone;
- stop relying on explicit source labels as proxies for city behavior;
- preserve no-true-capital prompt audit, source-disjoint holdout, direct
  geography controls, synthetic lookup controls, and answer-absent null rows;
- require at least 90 percent parseability plus both labels on non-holdout and
  holdout before any hidden-state work.
