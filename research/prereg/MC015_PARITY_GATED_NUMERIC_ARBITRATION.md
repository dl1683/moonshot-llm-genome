# MC015 Parity-Gated Numeric Arbitration Preregistration

Status: executed; behavior gate failed.

Date: 2026-07-01

## Objective

MC015 tests whether a prompt-local numeric lookup table can be controlled by a
learned factual gate without visible trusted/untrusted source-status text.

MC012 showed that explicit source-status labels can create a clean
local-versus-learned numeric behavior table. MC013 showed that removing that
status channel collapses the learned side. MC014 showed that inferred
calibration validity also collapses to prompt-local local-number answers.

MC015 changes the source/evaluation contract:

- the prompt contains a local lab-number table;
- the target atomic number is hidden from conflict prompts;
- no trusted/untrusted/reliable/unreliable/status lexeme is present in primary
  conflict prompts;
- the rule says whether the local table controls as a function of the queried
  element's standard atomic-number parity;
- expected local and expected atomic conflict rows are structurally balanced by
  split.

This is a behavior gate only. No hidden-state probe or intervention is allowed
unless the full behavior gate passes.

## Promotion Rule

Promote MC015 to hidden-signature screening only if the full generated behavior
run satisfies all of the following:

- structural prompt audit passes;
- 40 full sources are used with source-disjoint discovery/calibration/holdout
  splits;
- target and lure atomic numbers are hidden in conflict prompts;
- primary conflict prompts contain no source-status lexemes;
- expected local and expected atomic primary conflict labels are balanced by
  split;
- synthetic-key local lookup, familiar-element local lookup, real atomic-number
  recall, and answer-absent UNKNOWN controls pass;
- expected-local conflict rows select local numbers at least 85 percent of the
  time;
- expected-atomic conflict rows select atomic numbers at least 85 percent of the
  time;
- primary conflict expected-label correctness is at least 85 percent;
- non-holdout and holdout conflict outputs include both local and atomic/lure
  rows;
- candidate/output margins are reported.

## Death Rule

If direct controls pass but either expected-local or expected-atomic conflict
rows fail, MC015 is a diagnostic bridge death, not a probe substrate. The
allowed claim becomes a behavior-boundary claim about learned factual gates
under prompt-local table pressure.

## Containment Rule

If primary conflict rows are parseable and include both local and atomic
answers but do not follow the intended parity gate, preserve the result as a
separate diagnostic from MC014:

MC014 shows calibration-inferred source validity collapses to local table
answers. MC015 shows a learned factual gate can produce mixed outputs without
matching the intended control rule.

## Exported Diagnostics

- `PARITY_GATE_LABELS_BALANCED`
- `PARITY_GATE_NOT_FOLLOWED`
- `MC015_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`

## Executed Artifact

- runner:
  `code/mc015_parity_gated_numeric_arbitration.py`
- behavior status:
  `research/cards/MC015_PARITY_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`
- full behavior result:
  `results/cards/MC015/mc015_parity_gated_numeric_behavior_20260701T110321.json`
