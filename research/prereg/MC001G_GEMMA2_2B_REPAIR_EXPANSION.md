# MC001G Gemma 2 2B Repair Expansion Preregistration

Date: 2026-06-30

Status: preregistered for expanded behavior/matching substrate gate.

## Scope

- Card ID: `MC001G`
- Model: `google/gemma-2-2b`
- Stage: expanded repaired item-bank gate, no mechanism claim allowed
- Variant: `gemma_repair_expanded`
- Runner: `code/mc001_logit_smoke.py`
- Parent repair status: `research/cards/MC001G_GEMMA2_2B_REPAIR_STATUS.md`
- Latest sparse status: `research/cards/MC001G_GEMMA2_2B_SPARSE_DISCOVERY_STATUS.md`
- Planned manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_expanded_logit_raw_manifest.jsonl`
- Planned result directory: `results/cards/MC001G/`

## Rationale

MC001G has enough data to show a matched dense signature, but not enough to
support reliable intervention claims. The current matched holdout has only 14
wrong-hint rows, and activation replacement had to use nearest-bin donor
fallbacks for many rows. Canonical Gemma Scope sparse features also failed
promotion on this small holdout.

The next useful step is not another intervention. It is to expand the repaired
behavior substrate and check whether the same matching procedure yields a
larger exact-bin matched set.

## Fixed Design

Use the existing `gemma_repair` bank plus 64 additional stable objective
multiple-choice items:

- 128 total items;
- 32 correct answers per letter before model filtering;
- same seven conditions as the original repair gate:
  - `no_hint`;
  - `correct_hint`;
  - `wrong_disclaimed`;
  - `wrong_unsure`;
  - `wrong_direct`;
  - `wrong_high`;
  - `anti_wrong`;
- raw forced-choice next-token scoring;
- no free-generation evidence.

## Success Criteria

The expansion is useful only if it improves the future mechanism substrate:

- at least 72 clean items, where clean means `no_hint` and `correct_hint` are
  both truth-following;
- at least 12 clean items per correct answer letter;
- intermediate wrong-hint rows retain both truth-following and user-agreement
  labels;
- pre-hint-margin matching produces at least 80 matched wrong-hint rows total;
- matched holdout contains at least 24 rows;
- matched holdout bins are represented in discovery well enough to reduce or
  eliminate nearest-bin donor fallback for future activation patching.

## Failure Criteria

Do not proceed to sparse/path intervention if:

- the expanded bank mostly adds easy rows that stay truth-following under
  intermediate wrong hints;
- clean answer-letter balance fails;
- matched holdout remains too small;
- exact-bin donor coverage remains weak;
- output-margin or answer-letter baselines dominate again.

If the expansion fails, record it as a substrate failure and design a targeted
item-bank repair instead of running another final-token intervention.
