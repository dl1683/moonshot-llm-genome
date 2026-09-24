# MC007 Semi-Synthetic Familiar-Entity Lookup Route Closeout Status

Status: current MC007 route closed as diagnostic bridge; not signature-ready.

Date: 2026-07-01

## Diagnostic

Artifact diagnostic:

```text
mc007_route_closed_diagnostic_bridge
```

Canonical atlas diagnostic:

```text
MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE
```

## Verdict

MC007 tested the first bridge between MC005-style prompt-visible synthetic
lookup and MC006-style factual override.

The current MC007 route is closed as a diagnostic bridge, not a mechanism-card
route.

The route produced four useful behavior facts:

1. V1 showed that familiar country names can collapse into prompt-local lookup
   keys under terse table authority.
2. V2 showed that numeric authority pressure can create real-prior/lure
   contrast without true-capital prompt leaks.
3. V3 showed that target-only notes, compact notes, and stricter `Answer:`
   formatting do not repair the conflict substrate.
4. V4 showed that explicit `CITY`/`SOURCE` fields can decouple declared source
   from generated city behavior while controls fail.

That is enough to map the bridge boundary. It is not enough to start
hidden-state discovery or intervention.

## Evidence Chain

| Step | Result | Blocking Boundary |
| --- | --- | --- |
| V1 source-value baseline | `lookup_only` passed source-value behavior: 75/80 primary familiar rows were artificial-value answers, 0 were real-prior, 0 were lure, and answer-absent null was 40/40 `UNKNOWN` | no artificial-versus-real contrast |
| V2 authority dial | `numeric_dial` created contrast: 119 artificial, 24 real-prior, 6 lure, 51 unparsed primary rows; low-authority near miss had 15 artificial and 15 prior/lure rows | parseability 74.5 percent overall and 75.0 percent in the best panel; holdout prior/lure volume weak |
| V3 parse repair | direct controls stayed clean: real-world control 38/40 prior/lure, answer-absent null 40/40 `UNKNOWN` | selected `compact_note` primary table only 55.0 percent parseable with 62 artificial versus 4 prior/lure rows |
| V4 authority interface | explicit source/city format created some structure | selected controls failed: synthetic control 26/40 artificial, null 30/40 `UNKNOWN`, explicit source-city consistency 38/80, conflict consistency 12/80 |

## Interpretation

MC007 is not a failed bridge because nothing happened. It is valuable because
the failures are typed.

The bridge from synthetic lookup to factual override is governed by at least
four coupled surfaces:

- prompt-local table authority;
- real-world question authority;
- generated-answer parseability;
- consistency between declared source and generated city behavior.

The current route could move one or two of those surfaces at a time, but not
all four together. V1 had parseable lookup behavior without prior contrast. V2
had prior contrast without parseable, balanced generated answers. V3 repaired
controls without preserving contrast. V4 exposed source declarations as an
unsafe proxy for the city answer.

This means the next bridge attempt must change the behavior family or prompt
contract materially. Another prompt-only MC007 repair is not justified.

## Allowed Claims

- MC007 V1 is a clean source-value baseline for familiar entity names under a
  terse table-authoritative contract.
- MC007 V2 is the current best prior-pressure baseline because numeric
  authority can move generated labels and candidate margins without printing
  the true capital.
- MC007 V3 kills simple answer-slot and target-note parse repair.
- MC007 V4 kills explicit source declarations as a proxy for generated city
  behavior on this route.
- The current MC007 route is a diagnostic bridge showing prompt-authority and
  parseability boundaries between MC005 and MC006.

## Forbidden Claims

- MC007 is ready for hidden-state probing.
- MC007 found a familiar-entity knowledge-control signature.
- MC007 found a mechanism for factual recall or factual override.
- The V2 authority dial is causal control over parametric memory.
- The V4 source label is a valid intervention target or proxy label for the
  generated city answer.
- Another ordinary MC007 prompt-only repair remains a live mechanism-promotion
  route.

## Exported Rule

Future bridge work should preserve V1 and V2 as baselines, but must not repair
this same route indefinitely.

A new bridge family must, before hidden-state work:

- avoid true-capital prompt leaks;
- use generated answers, not only candidate scores;
- preserve source-disjoint holdouts;
- keep answer-absent null rows;
- pass direct geography and synthetic lookup controls;
- reach at least 90 percent parseability;
- include both artificial-value and real-prior/lure labels on non-holdout and
  holdout splits;
- audit candidate/output visibility before any hidden signature is promoted.

## Next Decision

Close MC007 V1-V4 as the first diagnostic bridge. The next bridge attempt
should be a new behavior family or prompt contract, not another response-format
repair inside the current route.
