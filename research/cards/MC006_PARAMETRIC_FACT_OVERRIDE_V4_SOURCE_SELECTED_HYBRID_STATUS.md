# MC006 Parametric Fact Override V4 Source-Selected Hybrid Status

Status: source-selected behavior substrate passed; narrow hidden-state discovery may proceed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V4_SOURCE_SELECTED_HYBRID.md`
- runner:
  `code/mc006_parametric_fact_override_v4_source_selected_hybrid.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json`
- result SHA256:
  `b3d153dde317161e2c028c7ff23aeeafd32d00a8b8d4dbb21415841d1360f32e`

## Verdict

MC006 V4 passes the preregistered source-selected behavior-substrate gate.

This is the first MC006 result that permits hidden-state discovery, but only
inside a narrow contract:

- V2-calibrated 23-source country-capital bank;
- V3 real-world paraphrase prompts;
- V4 strong fictional-codebook prompt;
- candidate-answer mean-logprob scoring;
- source-selected, prompt-bounded interpretation only.

The result is not a broad parametric-knowledge claim.

## Condition Summary

| Condition | Expected | True | Override | Lure | UNKNOWN | Expected-correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `direct_real_paraphrase` | true | 23 | 0 | 0 | 0 | 23/23 |
| `true_fact_paraphrase` | true | 23 | 0 | 0 | 0 | 23/23 |
| `false_claim_check` | true | 22 | 0 | 1 | 0 | 22/23 |
| `fictional_code_lookup` | override | 0 | 23 | 0 | 0 | 23/23 |

## Criteria

| Criterion | Result |
| --- | --- |
| direct-real true at least 20/23 | pass |
| true-fact true at least 22/23 | pass |
| false-claim true at least 20/23 | pass |
| false-claim override at most 3/23 | pass |
| fictional override at least 20/23 | pass |
| clean contrast sources at least 18 | pass |
| holdout clean contrasts at least 5 | pass |

## Source-Level Result

- clean source-level contrasts: 22/23;
- original holdout clean contrasts: 5/6;
- only non-clean source: Japan, where `false_claim_check` selected the lure
  answer `Osaka` instead of true `Tokyo`.

The diagnostic class is:

```text
source_selected_v4_substrate_passed
```

## Boundaries

V4 does not erase the prior failures:

- V1 failed because false reference-line authority dominated;
- V2 failed the full suite because direct real-world scoring was weak and
  `real_after_fiction` was broadly contaminated by the fictional rule;
- V3 failed because a weaker fictional-code prompt did not reliably induce the
  override.

The allowed next step is hidden-state signature discovery on the exact V4
contract. The first signature run must include output/logit-margin baselines,
source-split holdout evaluation, shuffled-label or row-label nulls, and an
explicit note that the substrate was selected from V2 behavior data.
