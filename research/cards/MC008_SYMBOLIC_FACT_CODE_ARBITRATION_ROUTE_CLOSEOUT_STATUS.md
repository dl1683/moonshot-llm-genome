# MC008 Symbolic Fact-Code Arbitration Route Closeout Status

Status: first symbolic bridge route closed as diagnostic; not signature-ready.

Date: 2026-07-01

## Diagnostic

Artifact diagnostics:

```text
symbolic_null_control_failed
symbolic_conflict_contrast_absent
```

Canonical atlas diagnostics:

```text
SYMBOLIC_NULL_CONTROL_FAILED
SYMBOLIC_CONFLICT_CONTRAST_WEAK
MC008_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE
```

## Verdict

MC008 tested a materially different bridge after MC007's open-city route
closed. It changed the answer interface from generated city names to compact
symbolic codes: chemical element symbols versus artificial symbol-like task
codes.

The first symbolic route is closed as a diagnostic bridge, not a
mechanism-card route.

V1 showed that compact symbolic answers can repair direct controls:

- synthetic code lookup reached 39/40 artificial-code rows;
- real-world chemical-symbol recall reached 39/40 real-symbol rows;
- generated answers were short and parseable enough for direct controls.

But V1 failed the behavior gate:

- answer-absent null rows were only 30/40 `UNKNOWN`;
- primary conflict parseability was 80.4 percent;
- primary conflict rows were table-dominant: 176 artificial-code rows versus
  8 real-symbol rows and 0 lure-symbol rows.

V2 then made the allowed materially different null/authority repair. It fixed
the null and direct controls:

- synthetic code lookup stayed clean at 39/40 artificial-code rows;
- real-world symbol recall improved to 40/40 real-symbol rows;
- answer-absent null rows became 40/40 `UNKNOWN`;
- primary conflict parseability rose to 96.25 percent.

But V2 failed the bridge:

- primary conflict rows were still 220 artificial-code rows versus only
  2 real-symbol rows and 0 lure-symbol rows;
- authority-0 rows produced only 1/40 real-symbol answers;
- non-holdout and holdout real/lure balance both failed.

## Interpretation

MC008 separates two boundaries that V1 had entangled.

The null boundary is repairable. A table-membership contract can stop the model
from inventing absent task codes without damaging the synthetic lookup or
real-symbol controls.

The conflict boundary is deeper. Once a matching artificial code is present in
the prompt table, Qwen3-1.7B overwhelmingly emits that code under this symbolic
interface, even when the prompt explicitly says ordinary chemistry should
control.

So compact symbolic answer classes do not solve the MC005-to-MC006 bridge. They
make the behavior easier to parse, but the prompt-local table still dominates
the generated answer under the tested contracts.

## Evidence

- V1 runner:
  `code/mc008_symbolic_fact_code_arbitration.py`
- V1 result:
  `results/cards/MC008/mc008_qwen3_1p7b_symbolic_fact_code_arbitration_behavior_20260701T061006.json`
- V1 SHA256:
  `9A586E0265529D4F47958DFAD09E9EF3DE2389A7A33594EB3E8B73631D8FA65A`
- V2 result:
  `results/cards/MC008/mc008_qwen3_1p7b_symbolic_fact_code_arbitration_v2_null_authority_repair_20260701T062249.json`
- V2 SHA256:
  `63A864747233F472A8A1C6C909EAE05B675157D6DBB1F28206DA83D8C8D5D5A0`

## Allowed Claims

- MC008 V1-V2 is a structurally clean symbolic-code bridge diagnostic on
  Qwen3-1.7B.
- Compact symbolic generated answers can make synthetic lookup and real-symbol
  recall controls clean.
- The answer-absent symbolic-code null boundary is repairable under an explicit
  table-membership contract.
- The first symbolic route fails because prompt-local task codes dominate
  artificial-versus-real conflict rows.

## Forbidden Claims

- MC008 is ready for hidden-state probing.
- MC008 found a symbolic knowledge-control surface.
- MC008 supports intervention or steering.
- Compact symbolic outputs solve the synthetic-lookup to factual-memory bridge.
- Prompt authority over the symbolic code table is a causal control surface.

## Exported Rule

Do not treat compact generated answer classes as sufficient bridge repair.

A future bridge must create real/lure conflict labels before probing. It should
not rely on another prompt-only rewrite around the same matching element-code
table. The next attempt should change at least one of:

- behavior family;
- answer representation;
- conflict construction;
- source visibility;
- evaluation interface.

## Next Decision

Close MC008 V1-V2 as the first symbolic bridge diagnostic.

The next bridge attempt should not be another ordinary MC008 prompt repair.
