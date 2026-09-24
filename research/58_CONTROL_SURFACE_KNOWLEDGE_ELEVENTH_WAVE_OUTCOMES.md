# Control-Surface Knowledge Eleventh-Wave Outcomes

Status: generated KSQ015 catalog-slash full-source outcome layer.

Generated data:

> `data/control_surface_knowledge_eleventh_wave_outcomes.json`

Builder:

> `code/control_surface_knowledge_eleventh_wave_outcomes.py`

Commands:

```powershell
python code\ksq015_catalog_slash_full_source_packet.py --write --write-prereg --write-status-card
python code\ksq015_catalog_slash_full_source_packet.py --score --score-candidates --full-run --write --write-status-card --local-files-only
python code\control_surface_knowledge_eleventh_wave_outcomes.py --write
python code\control_surface_knowledge_eleventh_wave_outcomes.py
python code\validate_control_surface_atlas.py
```

## Summary

- outcomes: `1`
- structural rows checked: `360`
- full behavior rows checked: `360`
- behavior-ready outcomes: `0`
- signature-screen licenses: `0`
- hidden-state licenses: `0`

## Diagnostic Chain

The catalog slash smoke survivor does not widen to a behavior substrate. The important finding is typed: catalog slash is local in no-bridge controls, but the underlying counted bridge surface is not reliable enough at full-source coverage.

| Step | Outcome | Result | Boundary |
| ---: | --- | --- | --- |
| 1 | `ksq015_catalog_slash_full_source_packet` | `CATALOG_SLASH_BRIDGE_POSITIVE_FAILED` | all catalog slash-only controls abstain 40/40 and raw answer_for reproduces 39/40, but exact bridge falls to 35/40 and source-disjoint holdout fails by one unparsed exact/reversed bridge row |

## Outcome

| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |
| --- | --- | --- | --- | --- | --- |
| `ksq015_catalog_slash_full_source_packet` | `tag_rows` | `CATALOG_SLASH_BRIDGE_POSITIVE_FAILED` | `false` | `false` | catalog controls clean, exact bridge not full-source reliable |

## Selected Panel Counts

- exact bridge: `{"evidence_answer": 35, "unparsed": 5}`
- raw answer_for alternate: `{"raw_answer_channel_reproduced": 39, "unparsed": 1}`
- catalog slash entity alternate: `{"abstain": 1, "evidence_answer": 36, "unparsed": 3}`
- catalog slash decoy alternate: `{"abstain": 3, "evidence_answer": 35, "unparsed": 2}`
- catalog slash reversed alternate: `{"evidence_answer": 35, "unparsed": 5}`
- catalog slash entity-only control: `{"control_abstain": 40}`
- catalog slash decoy-only control: `{"control_abstain": 40}`
- catalog slash reversed-only control: `{"control_abstain": 40}`
- query-only control: `{"control_abstain": 40}`

## Holdout Counts

- exact bridge: `{"evidence_answer": 7, "unparsed": 1}`
- raw answer_for alternate: `{"raw_answer_channel_reproduced": 8}`
- catalog slash entity alternate: `{"evidence_answer": 8}`
- catalog slash decoy alternate: `{"abstain": 1, "evidence_answer": 7}`
- catalog slash reversed alternate: `{"evidence_answer": 7, "unparsed": 1}`
- catalog slash entity-only control: `{"control_abstain": 8}`
- catalog slash decoy-only control: `{"control_abstain": 8}`
- catalog slash reversed-only control: `{"control_abstain": 8}`
- query-only control: `{"control_abstain": 8}`

## Failed Bridge Rows

| Source | Split | Panel | Label | First Line |
| --- | --- | --- | --- | --- |
| `nonce_08` | `discovery` | `exact_bridge` | `unparsed` | `K048T` |
| `nonce_08` | `discovery` | `catalog_slash_entity_alt` | `unparsed` | `K048T` |
| `nonce_08` | `discovery` | `catalog_slash_decoy_alt` | `abstain` | `UNKNOWN` |
| `nonce_08` | `discovery` | `catalog_slash_reversed_entity_alt` | `unparsed` | `K048T` |
| `nonce_12` | `calibration` | `exact_bridge` | `unparsed` | `K052T` |
| `nonce_12` | `calibration` | `catalog_slash_entity_alt` | `unparsed` | `K052T` |
| `nonce_12` | `calibration` | `catalog_slash_decoy_alt` | `unparsed` | `K052T` |
| `nonce_12` | `calibration` | `catalog_slash_reversed_entity_alt` | `unparsed` | `K052T` |
| `nonce_18` | `discovery` | `exact_bridge` | `unparsed` | `K058T` |
| `nonce_18` | `discovery` | `catalog_slash_entity_alt` | `abstain` | `UNKNOWN` |
| `nonce_18` | `discovery` | `catalog_slash_decoy_alt` | `abstain` | `UNKNOWN` |
| `nonce_18` | `discovery` | `catalog_slash_reversed_entity_alt` | `unparsed` | `K058T` |
| `nonce_21` | `holdout` | `exact_bridge` | `unparsed` | `K061T` |
| `nonce_21` | `holdout` | `catalog_slash_decoy_alt` | `abstain` | `UNKNOWN` |
| `nonce_21` | `holdout` | `catalog_slash_reversed_entity_alt` | `unparsed` | `K061T` |
| `nonce_27` | `calibration` | `exact_bridge` | `unparsed` | `K067T` |
| `nonce_27` | `calibration` | `catalog_slash_entity_alt` | `unparsed` | `K067T` |
| `nonce_27` | `calibration` | `catalog_slash_decoy_alt` | `unparsed` | `K067T` |
| `nonce_27` | `calibration` | `catalog_slash_reversed_entity_alt` | `unparsed` | `K067T` |

## Interpretation

KSQ015 kills the tempting read of KSQ014 as a ready catalog-slash substrate. The locality controls are clean, but the full-source positive bridge is not: the model sometimes returns code tokens instead of values. This is not a slash override; it is bridge extraction fragility under the current tag_rows contract.

## Forbidden Claims

- KSQ015 is behavior-ready.
- KSQ015 licenses hidden-state probing, intervention, or mechanism claims.
- Catalog slash notation is a reliable repair surface.
- The KSQ014 smoke result transfers cleanly to full-source coverage.

## Next Required Evidence

- Do not start a hidden-state signature screen from KSQ015.
- Treat the five code-token rows as bridge-extraction fragility, not slash override.
- Search for a bridge contract whose exact positive control clears full-source and holdout before re-testing catalog slash.
- Carry CATALOG_SLASH_BRIDGE_POSITIVE_FAILED into the failure taxonomy.
