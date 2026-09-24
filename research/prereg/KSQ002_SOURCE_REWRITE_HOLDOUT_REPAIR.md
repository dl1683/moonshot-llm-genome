# KSQ002 Source-Rewrite Holdout Repair

Status: predeclared single ordinary repair for the KSQ002 source-disjoint rewrite holdout boundary.

Runner:

> `code/ksq002_source_rewrite_holdout_repair.py`

Full behavior result:

> `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_full_behavior.json`

Status card:

> `research/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR_STATUS.md`

## Repair

Use one fixed template, `city_field_rewrite`, with the original KSQ002
panels and source split. The only intended change is the answer channel:
the prompt asks for the listed city name and asks the model not to repeat
the country.

## Promotion Rule

Promote only to behavior-substrate admission if the full 40-source run
passes baseline lookup, neutral rewrite, source deletion, query-only,
source-disjoint rewrite holdout, prompt audit, and candidate/output
margin-reporting gates.

## Kill Rule

Kill ordinary KSQ002 source-rewrite repair if this predeclared rerun still
misses source-disjoint holdout parseability or breaks deletion/query-only
controls.

## Forbidden Claims

- This repair is a mechanism card.
- This repair proves a source-channel, knowledge vector, or internal control surface.
- This repair supports intervention.
- A behavior pass permits a hidden-state claim without a later signature and intervention audit.
