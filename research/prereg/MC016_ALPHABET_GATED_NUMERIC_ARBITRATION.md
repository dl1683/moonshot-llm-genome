# MC016 Alphabet-Gated Numeric Arbitration Preregistration

Status: executed; behavior gate failed.

Date: 2026-07-01

## Objective

MC016 tests whether the numeric bridge can produce MC012-level local-versus-
learned contrast without trusted/untrusted source-status text by using a
visible non-status alphabetic gate.

MC012 showed that explicit source-status labels can create a clean mixed
local-versus-learned numeric behavior table. MC013 showed that direct
source-status ablation collapses the learned side. MC014 showed that
calibration-inferred source validity still collapses to local table answers.
MC015 showed that a learned atomic-number parity gate can create mixed outputs
without making the outputs follow the intended rule.

MC016 changes the source/evaluation contract:

- the prompt contains a local lab-number table;
- the target atomic number is hidden from conflict prompts;
- no trusted/untrusted/reliable/unreliable/status lexeme is present in primary
  conflict prompts;
- the rule says whether the local table controls as a function of the queried
  element's first-letter range, `A-M` versus `N-Z`;
- expected local and expected atomic conflict rows are structurally balanced by
  split;
- one template explicitly labels the neutral first-letter feature to test
  whether a visible non-status feature channel can rescue the atomic side.

This is a behavior gate only. Even if it passed, the first interpretation would
be prompt-visible operational control, not an internal knowledge mechanism.

## Promotion Rule

Promote MC016 only to a behavior-ready prompt-visible control note if the full
generated behavior run satisfies all of the following:

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
- expected-atomic conflict rows select atomic numbers at least 85 percent of
  the time;
- primary conflict expected-label correctness is at least 85 percent;
- non-holdout and holdout conflict outputs include both local and atomic/lure
  rows;
- candidate/output margins are reported.

Even under promotion, MC016 would not become signature-ready because the
alphabet gate is prompt-visible by design.

## Death Rule

If direct controls pass but expected-atomic conflict rows fail, MC016 is a
diagnostic bridge death, not a probe substrate. The allowed claim becomes a
behavior-boundary claim about prompt-local table dominance under non-status
operational gates.

## Containment Rule

If the explicit feature-labeled template improves parseability or mixture but
still fails expected-atomic rows, preserve it as a separate diagnostic:

MC016 then says that the bridge failure is not merely inability to infer the
alphabet feature from the element string. Even a visible neutral feature channel
does not reproduce MC012's learned-atomic side under this prompt contract.

## Exported Diagnostics

- `ALPHABET_GATE_LABELS_BALANCED`
- `ALPHABET_GATE_LOCAL_COLLAPSE`
- `FEATURE_LABEL_GATE_DID_NOT_RESCUE`
- `MC016_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`

## Executed Artifact

- runner:
  `code/mc016_alphabet_gated_numeric_arbitration.py`
- behavior status:
  `research/cards/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`
- full behavior result:
  `results/cards/MC016/mc016_alphabet_gated_numeric_behavior_20260701T113651.json`
