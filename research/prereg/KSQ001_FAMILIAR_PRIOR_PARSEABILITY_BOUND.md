# KSQ001 Familiar-Prior Parseability Bound

Status: predeclared second-wave bounded repair-or-closeout.

Runner:

> `code/ksq001_familiar_prior_parseability_bound.py`

Full behavior result:

> `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_full_behavior.json`

Status card:

> `research/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND_STATUS.md`

## Templates

- `compact_original_replay`
- `compact_begin_city`
- `compact_city_name_first`

## Promotion Rule

Admit behavior only if a full 40-source run reaches conflict
parseability >= 0.90 while preserving local artificial lookup,
semantic-prior direct recall, answer-absent nulls, conflict mixture,
source-disjoint holdout mixture, prompt audit, and candidate/output
margin reporting.

## Kill Rule

Kill ordinary KSQ001 repair if parseability improves only by
collapsing to local lookup, semantic-prior recall, or UNKNOWN, or if
direct controls/nulls regress.

## Forbidden Claims

- This repair is a mechanism card.
- Parser repair alone is behavior repair.
- A behavior pass is a hidden-state claim or intervention.
- Familiar-prior pressure is a knowledge-control surface.
