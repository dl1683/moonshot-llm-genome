# MC007 Semi-Synthetic Familiar-Entity Lookup V4 Authority-Interface Status

Status: authority-interface redesign failed the control panel and exposed a
source-declaration/city-generation mismatch.

Date: 2026-07-01

## Artifact

- V4 runner:
  `code/mc007_semi_synthetic_familiar_entity_lookup_v4_authority_interface.py`
- full result:
  `results/cards/MC007/mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_v4_authority_interface_20260701T020837.json`
- full result SHA256:
  `927D3BCBB13DDB0A47DF981C818C60F8600936831DAE65A1E8ED8EB806FAB738`

## Verdict

MC007 V4 tested a materially different response interface after V3 killed
answer-slot repair. Instead of asking for a bare city, V4 asked for both:

```text
CITY: one city name, or UNKNOWN
SOURCE: TASK, MEMORY, or UNKNOWN
```

The goal was to see whether explicit source reporting could preserve
artificial-versus-real contrast while making the generated answer parseable.

The artifact diagnostic is:

```text
control_panel_failed
```

The result is not signature-ready:

```text
passed = false
signature_ready = false
```

The useful finding is sharper than another parse failure:

> Source declarations and city generation came apart. The model could emit
> `TASK` or `MEMORY`-like source labels while the city field followed different
> pressure. This makes explicit source labels unsafe as a proxy for the
> generated factual answer.

## Structural Result

The full run was structurally clean:

- model: `Qwen/Qwen3-1.7B`;
- 40 country sources;
- 2 templates: `city_then_source`, `source_then_city`;
- 7 panels per source/template;
- 560 generated rows;
- source-disjoint split;
- no duplicate record ids;
- no candidate collisions;
- no true-capital prompt leaks;
- selected prompt audit passed.

## Gate Result

| Criterion | Result |
| --- | --- |
| structural checks | pass |
| full source count is 40 | pass |
| real-world control prior/lure at least 85 percent | pass: 38/40 |
| synthetic control artificial at least 90 percent | fail: 26/40 on selected template |
| answer-absent null unknown at least 95 percent | fail: 30/40 on selected template |
| explicit authority parseability at least 90 percent | fail: 57/80 = 71.25 percent |
| explicit task artificial at least 80 percent | fail: 17/40 on selected template |
| explicit real prior/lure at least 80 percent | fail: 22/40 on selected template |
| conflict contrast present | pass |
| within-conflict behavior gate | fail |
| signature ready | false |

The selected template was `city_then_source` for both explicit and conflict
selection because it had the best control floor. The alternative
`source_then_city` looked better on explicit task/real labels, but collapsed
on real-world and null controls.

## Selected Template: City Then Source

### Control Panels

| Panel | Artificial | Prior/Lure | Unknown | Unparsed | Parseable | Source-City Consistent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic control lookup | 26 | 0 | 0 | 14 | 65.0 percent | 65.0 percent |
| real-world control | 0 | 38 | 0 | 2 | 95.0 percent | 77.5 percent |
| answer-absent null | 0 | 0 | 30 | 10 | 75.0 percent | 75.0 percent |

Direct real-world geography remained mostly clean, but synthetic lookup and
answer-absent null controls failed under the two-field response interface.

### Explicit Authority Panels

Across `task_query_with_note` and `real_query_with_note`:

- 80 rows;
- 20 artificial-value rows;
- 31 real-prior rows;
- 5 lure rows;
- 1 unknown row;
- 23 unparsed rows;
- city parseability: 57/80 = 71.25 percent;
- source parseability: 65/80 = 81.25 percent;
- source-city consistency: 38/80 = 47.5 percent.

This is enough to show a prompt-visible task-versus-real distinction, but not
enough to form a reliable behavior substrate. The city field and source field
often disagree.

### Ambiguous Conflict Panels

Across `ambiguous_authority_30` and `ambiguous_authority_0`:

- 80 rows;
- 4 artificial-value rows;
- 29 real-prior rows;
- 5 lure rows;
- 7 unknown rows;
- 35 unparsed rows;
- city parseability: 45/80 = 56.25 percent;
- source parseability: 57/80 = 71.25 percent;
- source-city consistency: 12/80 = 15.0 percent.

The conflict panels contain contrast, but it is weak, imbalanced, and not
parseable enough for hidden-state work. The source labels are especially
unreliable: 38/80 conflict rows declared `TASK`, while only 4/80 city fields
were artificial task values.

## Alternative Template: Source Then City

`source_then_city` is an instructive negative control.

Explicit task/real rows looked superficially better:

- 74/80 city-parseable;
- 34 artificial-value rows;
- 38 real-prior rows;
- 2 lure rows.

But the same template broke the controls:

- real-world control: only 3/40 real-prior city fields, 37/40 unparsed;
- answer-absent null: only 4/40 `UNKNOWN`, 36/40 unparsed;
- synthetic control: only 14/40 artificial task-city fields.

This is why the V4 selector correctly refused to promote `source_then_city`.
The template can produce source-like text, but it does not preserve behavior
locality across controls.

## Interpretation

V4 adds a new diagnostic boundary:

```text
SOURCE_DECLARATION_CITY_MISMATCH
```

The mismatch matters because an apparently clean source-choice behavior could
be a false substrate. If the model says `SOURCE: TASK` but writes a real
capital, or says `SOURCE: MEMORY` while writing the prompt-local task city, a
probe over source labels would not be measuring the generated behavior we care
about.

MC007 now has four bridge facts:

1. V1: familiar countries can collapse into prompt-local lookup keys.
2. V2: authority pressure can create real-prior/lure contrast, but the table is
   parse-fragile.
3. V3: stricter answer-slot prompting does not repair the conflict panel.
4. V4: explicit source reporting creates more structure, but source labels,
   city fields, and controls do not stay aligned.

This route still teaches the atlas, but it is not a mechanism-card route.

## Allowed Claims

- MC007 V4 was structurally clean and preserved source-disjoint holdouts.
- V4 created some artificial-versus-prior city contrast under explicit
  authority fields.
- V4 failed the behavior substrate because selected-template controls,
  parseability, and source-city consistency failed.
- Explicit source declarations are not reliable substitutes for city-answer
  behavior on this task.
- MC007 remains blocked before hidden-state probing.

## Forbidden Claims

- MC007 V4 is ready for hidden-state probing.
- The V4 source label is a reliable authority-decision target.
- A source-choice probe would be a valid proxy for generated city behavior on
  this artifact.
- The V4 authority interface repaired MC007.
- MC007 has found a familiar-entity knowledge-control surface.

## Next Step

Do not run hidden-state discovery on MC007 V4.

At this point the current MC007 route should be treated as a diagnostic bridge
unless a genuinely different behavior family is introduced. The strongest
export is the typed failure:

```text
prompt authority can move the bridge, but generated source declarations,
city answers, parseability, and null controls do not remain coupled under the
current response interfaces.
```
