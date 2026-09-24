# MC006 Parametric Fact Override V2 Repair Status

Status: behavior substrate failed; false-claim audit improved, but fictional-rule contamination remains broad.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V2_REPAIR.md`
- runner:
  `code/mc006_parametric_fact_override_v2_repair.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json`
- result SHA256:
  `7ae7ae5a1d1a9cc1a36d7ac9684432b913dfc2bd9ecfb1ec556de76f66ebca8b`

## Verdict

MC006 V2 does not pass the behavior-substrate gate. Do not start hidden-state
signature discovery from the full V2 prompt suite.

The repair succeeded on two controls:

- verified true context selected the true capital on 40/40 rows;
- fictional task-local city-code override selected the override city on 40/40
  rows.

The false-claim audit almost repaired the V1 reference-line failure:

- `false_claim_audit` selected the true answer on 29/40 rows;
- `false_claim_audit` selected the override on 8/40 rows, exactly at the
  preregistered override ceiling.

But the full substrate still failed:

- direct real-world answering selected the true answer on only 25/40 rows;
- `real_after_fiction` selected the true answer on only 9/40 rows;
- `real_after_fiction` selected the override on 21/40 rows;
- strict clean source-level contrasts were only 9/40, with 1/8 holdout.

The diagnostic class is:

```text
parametric_fact_failed
```

The stricter interpretation is that V2 contains two distinct failures:

1. candidate scoring still often prefers salient non-capital cities over true
   capitals in direct real-world prompts;
2. once a fictional codebook rule is mentioned, the model often keeps following
   it even when explicitly told to ignore it for geography.

## Condition Summary

| Condition | Expected | True | Override | Lure | UNKNOWN | Expected-correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `direct_real` | true | 25 | 5 | 10 | 0 | 25/40 |
| `true_fact` | true | 40 | 0 | 0 | 0 | 40/40 |
| `false_claim_audit` | true | 29 | 8 | 3 | 0 | 29/40 |
| `fictional_override` | override | 0 | 40 | 0 | 0 | 40/40 |
| `real_after_fiction` | true | 9 | 21 | 10 | 0 | 9/40 |

## Criteria

| Criterion | Result |
| --- | --- |
| direct-real true at least 32/40 | fail |
| true-fact true at least 36/40 | pass |
| false-claim-audit true at least 30/40 | fail |
| false-claim-audit override at most 8/40 | pass |
| fictional-override override at least 24/40 | pass |
| real-after-fiction true at least 30/40 | fail |
| real-after-fiction override at most 8/40 | fail |
| clean contrast sources at least 16 | fail |
| holdout clean contrasts at least 6 | fail |

## Useful Lead

V2 produced a narrower lead that should not be promoted without a new
preregistration:

- 23/40 sources passed the three-way contrast:
  `direct_real=true_answer`, `false_claim_audit=true_answer`, and
  `fictional_override=override_answer`;
- 6 of those sources were in holdout.

This suggests a possible V3 repair: treat V2 as a calibration run, preregister a
source-selected capital-fact substrate, and test source-disjoint or
paraphrase-holdout variants before hidden-state discovery. The V3 design must
either remove `real_after_fiction` from the promotion gate as a documented
side-effect boundary, or replace it with a locality control that does not put a
fresh fictional mapping in the same prompt as the real-world question.

## Interpretation

V2 is a negative repair, not a mechanism-card substrate. It shows that Qwen3-1.7B
can follow fictional city-code mappings and can reject many false capital
claims, but this prompt/scoring interface still does not cleanly expose a
reliable parametric-fact control surface.
