# MC001G Gemma 2 2B Format-Control Status

Status: complete. Option-position counterbalancing improved the substrate, but
the current letter-choice Gemma route still fails the preregistered reliability
bar for renewed hidden-state work.

Date: 2026-06-30

## Artifacts

Format control v1:

- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_format_control_audit.py`
- variant: `gemma_repair_permuted`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_FORMAT_CONTROL.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_format_control_logit_raw_manifest.jsonl`
- manifest SHA256: `12a5004a7d23ede750bef563c17e9500c199f89f9c9fd58de6ec0bab6f602f32`
- result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_logit_raw_gemma_repair_permuted_20260630T133240.json`
- result SHA256: `06134e64294d9c5c7688b63adcfc1afe917cc47037c8503b85e47b65b7b82872`
- audit result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_source_audit_20260630T133257.json`
- audit SHA256: `a5864d75d5ca04d26022a5aa726928b5f67fce2085bb44cbe50e4455155dfdaf`

Format control V2:

- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_format_control_audit.py`
- variant: `gemma_repair_permuted_expanded`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_FORMAT_CONTROL_V2.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_format_control_v2_logit_raw_manifest.jsonl`
- manifest SHA256: `9b8870f41283a5ae0e18f44ddb2c28fcba4de52ac8bb30567d9315aa70c9678f`
- result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_v2_logit_raw_gemma_repair_permuted_expanded_20260630T133648.json`
- result SHA256: `612cbae98b97ae268a91cd37d63c4297807683d9fe4337b569e76ff5556b938a`
- audit result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_v2_source_audit_20260630T133701.json`
- audit SHA256: `3df075c01268f777a1e16982da16e2d4d66e855d302138a5deee7c1950e45ccf`

Common configuration:

- model: `google/gemma-2-2b`
- render mode: `raw`
- conditions: unchanged `CONDITIONS_GEMMA_REPAIR`
- split mode: source-group split
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- strict match: item-level `no_hint` margin bin plus correct answer letter

## Question

Can answer-position counterbalancing repair the MC001G forced-choice substrate
enough to resume hidden-state discovery?

## Gate Verdict

No.

Counterbalancing fixed the worst matched-label answer-letter skew, but neither
format-control run passed all preregistered reliability checks. The current
letter-choice Gemma MC001G route should stop here.

Write this as:

> MC001G format control shows that option-position balancing helps, but the
> letter-choice substrate still lacks exact source-disjoint coverage for a
> reliable intervention test.

## Format Control V1

V1 used the 64 original repaired source questions and generated four option
permutations for each source.

Structural manifest checks passed:

| Metric | Result |
| --- | ---: |
| source groups | 64 |
| items | 256 |
| records | 1,792 |
| correct A/B/C/D items | 64 each |
| wrong A/B/C/D items | 64 each |

Behavior and audit:

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 151 | at least 96 | pass |
| min clean items per letter | 22 | at least 18 | pass |
| format-complete clean sources | 12 | at least 16 | fail |
| primary truth rows | 132 | at least 60 | pass |
| primary agreement rows | 159 | at least 60 | pass |
| bin-only matched rows | 168 | at least 96 | pass |
| strict matched rows | 98 | at least 64 | pass |
| strict matched holdout rows | 26 | at least 24 | pass |
| max single-letter share of either strict matched label | 22/49 | at most 50% | pass |
| every strict holdout bin-letter key covered in discovery | no | yes | fail |

Strict matched correct-letter label counts:

| Correct Letter | Truth | Agreement |
| --- | ---: | ---: |
| A | 22 | 22 |
| B | 10 | 10 |
| C | 13 | 13 |
| D | 4 | 4 |

V1 was a real improvement over targeted repair because strict matching removed
the A-correct agreement dominance. It still failed source-complete coverage and
exact strict holdout support.

## Format Control V2

V2 used 128 source questions: the original repaired bank plus the broad
expansion bank.

Structural manifest checks passed:

| Metric | Result |
| --- | ---: |
| source groups | 128 |
| items | 512 |
| records | 3,584 |
| correct A/B/C/D items | 128 each |
| wrong A/B/C/D items | 128 each |

Behavior and audit:

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 270 | at least 220 | pass |
| min clean items per letter | 36 | at least 32 | pass |
| format-complete clean sources | 24 | at least 24 | pass |
| primary truth rows | 236 | at least 100 | pass |
| primary agreement rows | 293 | at least 100 | pass |
| bin-only matched rows | 276 | at least 160 | pass |
| strict matched rows | 174 | at least 128 | pass |
| strict matched holdout rows | 54 | at least 40 | pass |
| min strict matched rows per letter-label cell | 7 | at least 8 | fail |
| max single-letter share of either strict matched label | 35/87 | at most 50% | pass |
| every strict holdout bin-letter key covered in discovery | no | yes | fail |

Strict matched correct-letter label counts:

| Correct Letter | Truth | Agreement |
| --- | ---: | ---: |
| A | 35 | 35 |
| B | 18 | 18 |
| C | 27 | 27 |
| D | 7 | 7 |

Strict holdout bin-letter keys lacking discovery support:

- `1|B`
- `1|D`
- `3|C`

V2 is the strongest MC001G behavior substrate so far, but it still fails the
precommitted reliability bar.

## Diagnosis

Answer-position counterbalancing is useful but not enough.

The V2 strict matched set is much less confounded than the targeted-repair set:
A-correct rows no longer dominate matched agreement after strict matching. But
the model still has a position-dependent no-hint failure pattern. D-correct
items remain sparse after clean filtering and strict matching, and exact
source-disjoint discovery support is still missing for some holdout bin-letter
cells.

This means a dense, sparse, path, or activation-replacement intervention on the
current letter-choice format could still succeed or fail for the wrong reason.

## Decision

Do not resume hidden-state discovery or intervention on the current MC001G
letter-choice substrate.

Close this Gemma letter-choice route as:

> repaired behavior, matched signatures, failed interventions, sparse-promotion
> failure, and format-control failure.

The next MC001G-style mechanism attempt should change the behavior interface
more deeply: use non-letter answers, pairwise choices, or generated-answer
grading, then rebuild the signature and intervention gates from that substrate.
