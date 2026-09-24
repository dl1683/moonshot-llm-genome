# KSQ015 Catalog Slash Full-Source Packet

Status: KSQ014 diagnostic follow-up.

Runner:

> `code/ksq015_catalog_slash_full_source_packet.py`

Artifacts:

> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_first_run.json`

> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_smoke_limit10.json`

> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_full_behavior.json`

## Claim Under Test

KSQ014 showed that catalog-labeled slash text survived a 10-source
locality smoke, while bare slash failed. KSQ015 removes bare slash and
tests whether catalog slash alone survives all 40 sources and the
source-disjoint holdout.

## Promotion Rule

Promote only to behavior-substrate status if exact bridge answers, raw
answer_for remains an active positive control, every catalog slash
bridge panel answers the counted bridge rather than the alternate, every
catalog slash-only no-bridge control abstains, query-only abstains,
selected prompt audit passes, source-disjoint holdout passes, and
candidate margins are reported on the full 40-source run.

A promotion licenses only a later margin-matched signature screen. It
does not license an intervention or mechanism claim.

## Kill / Boundary Rule

If raw answer_for no longer reproduces, the packet is not anchored to the
KSQ011-KSQ014 pressure. If catalog slash bridge panels override the
bridge, export catalog slash override. If catalog slash-only controls
reproduce a value, export catalog slash locality leakage. If source-
disjoint holdout fails while aggregate panels pass, export holdout
fragility.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- A behavior pass would not be a hidden-state or intervention result.
