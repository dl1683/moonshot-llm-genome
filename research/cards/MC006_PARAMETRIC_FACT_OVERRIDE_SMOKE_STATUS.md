# MC006 Parametric Fact Override Smoke Status

Status: behavior substrate failed; context-authority artifact dominates.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_SMOKE.md`
- runner:
  `code/mc006_parametric_fact_override_smoke.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_smoke_20260630T221737.json`
- result SHA256:
  `75c8f1b4a230ffff92216cfd1a8ca71f27c816ff7ed9059fdc07a7f2dbaa9b9e`

## Verdict

MC006 does not pass the behavior-substrate gate. Do not start hidden-state
signature discovery on this prompt/scoring interface.

The model showed two strong but incompatible behaviors:

- true supporting context repaired factual answering to 40/40;
- explicit task-local overwrite was followed on 39/40 rows.

But the controls failed:

- no-context true-answer selection was only 26/40;
- irrelevant-context true-answer selection was only 27/40;
- `mistake_context` selected the false override on 40/40 rows;
- clean source-level contrasts were 0/40.

The diagnostic class is:

```text
parametric_fact_failed
```

The stricter interpretation is broader than that label: the run failed both the
parametric-fact floor and the context-locality floor.

## Condition Summary

| Condition | Expected | True | Override | Lure | UNKNOWN | Expected-correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `no_context` | true | 26 | 13 | 1 | 0 | 26/40 |
| `true_context` | true | 40 | 0 | 0 | 0 | 40/40 |
| `irrelevant_context` | true | 27 | 9 | 4 | 0 | 27/40 |
| `task_override` | override | 0 | 39 | 1 | 0 | 39/40 |
| `mistake_context` | true | 0 | 40 | 0 | 0 | 0/40 |

## Criteria

| Criterion | Result |
| --- | --- |
| no-context true at least 30/40 | fail |
| true-context true at least 30/40 | pass |
| irrelevant-context true at least 30/40 | fail |
| mistake-context true at least 30/40 | fail |
| task-override override at least 16/40 | pass |
| mistake-context override at most 8/40 | fail |
| clean contrast sources at least 16 | fail |
| holdout clean contrasts at least 6 | fail |

## Interpretation

MC006 was designed to find sources where the model chooses:

- the parametric real-world fact with no context;
- the task-local false mapping only when explicitly told to use task-local
  mappings;
- the real-world fact when warned that the false line may be a mistake.

It found none.

This does not mean Qwen3-1.7B lacks parametric capital knowledge. `true_context`
made every row correct, and several no-context rows were correct. The failure is
that this candidate-scoring prompt lets the reference line dominate even when
the instruction says the line may be wrong. The behavior is therefore not a
clean parametric-knowledge/context-overwrite substrate.

## Next Step

The next MC006 repair should change the interface before hidden-state work:

1. separate the real-world and task-local prompts more aggressively;
2. avoid city-overwrite candidates that are more salient city completions than
   the true capital in no-context rows;
3. add a stricter real-world-only control with no reference-line syntax;
4. optionally test a generated-answer variant, but only with a parse and
   answer-shape audit.

No mechanism-card claim is allowed from MC006 V1.
