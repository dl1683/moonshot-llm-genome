# MC001G Gemma 2 2B Pairwise Text Status

Status: complete. Pairwise answer-text scoring removed literal `A`/`B`/`C`/`D`
outputs and fixed exact source-disjoint holdout coverage, but the behavior
remained pair-order confounded. The pairwise interface does not reopen
hidden-state discovery or intervention work.

Date: 2026-06-30

## Artifacts

Pairwise text v1:

- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_pairwise_text_audit.py`
- variant: `gemma_pairwise_text`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_PAIRWISE_TEXT.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_pairwise_text_logit_raw_manifest.jsonl`
- manifest SHA256: `8210aa36cf3c174bfdb2cfb7c96a09c3a79ddbdbdebe203e0d1cdfcdb6835f1a`
- result: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_logit_raw_gemma_pairwise_text_20260630T135443.json`
- result SHA256: `c5bbacdb1a86f5e39b6d6da6032dfe86734d362c4f34d324351642aa5d56fef3`
- audit result: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_source_audit_20260630T135457.json`
- audit SHA256: `c87ebb4e25c813d1b087e115a17db1accc70c09b201d372084b3ebb12f3c76b2`

Pairwise text V2:

- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_pairwise_text_audit.py`
- variant: `gemma_pairwise_text_v2`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_PAIRWISE_TEXT_V2.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_pairwise_text_v2_logit_raw_manifest.jsonl`
- manifest SHA256: `8fa7bf26db6a4ed3946723f81b383804dba826859995d32f7e9ed5a3322e51fd`
- result: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_v2_logit_raw_gemma_pairwise_text_v2_20260630T140430.json`
- result SHA256: `65ab128ce8d0452b78591f601580397fef75ae7a013bf065f303643e6d1f1aa5`
- audit result: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_v2_source_audit_20260630T140448.json`
- audit SHA256: `f8c4e7c4971cb3444c96ed22a358fbc7e61edc6b67cfd3b017ead95d7567e182`

Common configuration:

- model: `google/gemma-2-2b`
- render mode: `raw`
- scoring: sequence logprob of the candidate answer text, using the better mean
  logprob across no-leading-space and leading-space variants
- source split: source-group split, so both pair orders for a source question
  stay in discovery or holdout together
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- strict match: item-level `no_hint` margin bin plus pair order
- margin bin width: `0.5`

## Question

Can a non-letter pairwise answer-text interface repair the MC001G behavior
substrate enough to restart hidden-state discovery?

## Gate Verdict

No.

Pairwise answer-text scoring is a useful control improvement. It removes the
literal letter-output surface and produces exact source-disjoint holdout
coverage. It still fails the preregistered order-balance guards. A hidden-state
signature or intervention on this substrate could still be measuring where the
answer text appears in the pair, not the truth-versus-agreement behavior.

Write this as:

> MC001G pairwise text fixed literal letter outputs and exact holdout support,
> but failed pair-order balance. It is an interface-control failure, not a
> mechanism substrate.

## Pairwise Text V1

V1 used the original repaired source bank plus the broad expansion bank. It
dropped source questions whose correct or weak-wrong answer text was exactly
`A`, `B`, `C`, or `D`, then generated two pair orders for each retained source.

Structural manifest checks passed:

| Metric | Result |
| --- | ---: |
| retained source groups | 125 |
| pairwise items | 250 |
| records | 1,750 |
| `cw` items | 125 |
| `wc` items | 125 |
| exact `A`/`B`/`C`/`D` answer texts | 0 |

Behavior and audit:

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 226 | at least 150 | pass |
| clean `cw` items | 108 | at least 60 | pass |
| clean `wc` items | 118 | at least 60 | pass |
| pair-complete clean sources | 105 | at least 60 | pass |
| primary truth rows | 104 | at least 80 | pass |
| primary agreement rows | 348 | at least 80 | pass |
| bin-only matched rows | 156 | at least 120 | pass |
| strict matched rows | 124 | at least 96 | pass |
| strict matched holdout rows | 42 | at least 30 | pass |
| min pair-order label cell | 18 | at least 20 | fail |
| max single-order share of either strict label | 44/62 | at most 65% | fail |
| every strict holdout bin-order key covered in discovery | yes | yes | pass |

Strict matched pair-order label counts:

| Pair Order | Truth | Agreement |
| --- | ---: | ---: |
| `cw` | 18 | 18 |
| `wc` | 44 | 44 |

V1 was the first MC001G branch to pass exact strict holdout bin-order coverage,
but it failed the preregistered pair-order balance guard.

## Pairwise Text V2

V2 added the targeted and targeted-V2 source banks to test whether the remaining
order skew was a sparse-coverage problem.

Structural manifest checks passed:

| Metric | Result |
| --- | ---: |
| retained source groups | 253 |
| pairwise items | 506 |
| records | 3,542 |
| `cw` items | 253 |
| `wc` items | 253 |
| exact `A`/`B`/`C`/`D` answer texts | 0 |

Behavior and audit:

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 464 | at least 320 | pass |
| clean `cw` items | 223 | at least 140 | pass |
| clean `wc` items | 241 | at least 140 | pass |
| pair-complete clean sources | 218 | at least 140 | pass |
| primary truth rows | 178 | at least 160 | pass |
| primary agreement rows | 750 | at least 160 | pass |
| bin-only matched rows | 288 | at least 240 | pass |
| strict matched rows | 256 | at least 192 | pass |
| strict matched holdout rows | 92 | at least 60 | pass |
| min pair-order label cell | 34 | at least 40 | fail |
| max single-order share of either strict label | 94/128 | at most 65% | fail |
| every strict holdout bin-order key covered in discovery | yes | yes | pass |

Strict matched pair-order label counts:

| Pair Order | Truth | Agreement |
| --- | ---: | ---: |
| `cw` | 34 | 34 |
| `wc` | 94 | 94 |

V2 increased the matched set substantially and preserved exact holdout
bin-order coverage, but the larger source bank did not fix pair-order
dominance. `wc` supplied 94/128 rows for each strict matched label.

## Diagnosis

The behavior is now cleaner than the letter-choice substrate, but not reliable
enough for a mechanism search.

The scoring surface no longer asks for one of four letter tokens, and source
questions are split cleanly between discovery and holdout. Those are real
improvements. The failure is that answer order still predicts membership in the
strict matched set too strongly. Any later internal signature might be a
signature of pair layout, answer-position salience, or choice ordering rather
than a truth-versus-agreement control surface.

## Decision

Do not resume hidden-state discovery, sparse-feature discovery, path
localization, activation patching, or steering on the MC001G pairwise answer
text interface.

Close this branch as:

> repaired behavior, matched signatures, failed interventions,
> sparse-promotion failure, format-control failure, and pairwise-interface
> failure.

The next MC001G-style attempt should use generated-answer grading or an
answer-text interface without displayed pair choices, then rebuild the
signature, intervention, and reliability gates from that substrate.
