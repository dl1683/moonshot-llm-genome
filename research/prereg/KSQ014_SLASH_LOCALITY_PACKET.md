# KSQ014 Slash Locality Packet

Status: KSQ013 diagnostic follow-up.

Runner:

> `code/ksq014_slash_locality_packet.py`

Artifacts:

> `results/cards/KSQ014_SLASH_LOCALITY_PACKET/ksq014_slash_locality_packet_first_run.json`

> `results/cards/KSQ014_SLASH_LOCALITY_PACKET/ksq014_slash_locality_packet_smoke_limit10.json`

> `results/cards/KSQ014_SLASH_LOCALITY_PACKET/ksq014_slash_locality_packet_full_behavior.json`

## Claim Under Test

KSQ013 found that `CATALOG_NOTE entity / alternate` preserved the
counted bridge at `9/10` in smoke but lacked a matching slash-only
control. KSQ014 tests matched slash bridge and slash-only no-bridge
panels directly.

## Promotion Rule

Promote only to behavior-substrate candidate status if exact bridge
answers, raw answer_for remains an active positive control, every slash
bridge panel answers the counted bridge rather than the alternate, every
slash-only no-bridge control abstains, query-only abstains, selected
prompt audit passes, source-disjoint holdout passes, and margins are
reported.

## Kill / Boundary Rule

If raw answer_for no longer reproduces, the packet is not anchored to the
KSQ011-KSQ013 pressure. If slash bridge panels override the bridge,
export slash override. If slash-only controls reproduce a value, export
slash locality leakage. If slash bridge panels lose the target without
selecting the alternate, export bridge loss.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden unless a later full behavior
  run and margin report admit only a signature screen.
