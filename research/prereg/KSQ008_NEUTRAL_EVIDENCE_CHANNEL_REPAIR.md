# KSQ008 Neutral-Evidence Channel Repair

Status: KSQ007B diagnostic follow-up.

Runner:

> `code/ksq008_neutral_evidence_channel_repair.py`

Artifacts:

> `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_first_run.json`

> `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_smoke_limit10.json`

> `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_full_behavior.json`

## Claim Under Test

KSQ007B localized the leak to answer-bearing `answer_for(entity)=value`
syntax. This repair tests whether counted evidence can move to a
neutral row grammar while forbidden answer_for strings, uncounted
neutral rows, quoted neutral rows, wrong-schema counted rows, absent
rows, unrelated rows, conflicts, and query-only controls abstain.

## Promotion Rule

Promote only to behavior-substrate candidate status if exact neutral
evidence answers, neutral evidence wins against a forbidden bare
answer_for alternate, every abstention/control panel passes, selected
prompt audit passes, source-disjoint holdout passes, and margins are
reported. Hidden-state claims remain forbidden.

## Kill / Boundary Rule

If exact neutral evidence does not answer, close this repair route as
`NEUTRAL_EVIDENCE_POSITIVE_FAILED`. If answer_for controls still
reproduce values, export `ANSWER_FOR_SYNTAX_SURVIVES_NEUTRAL_REPAIR`.
If uncounted or quoted neutral rows reproduce values, export a neutral
channel locality failure. Treat the typed failure as the datum.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden unless a full behavior run and
  margin report later admit only a signature screen.
