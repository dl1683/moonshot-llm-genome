# MC012 Reliability-Labeled Numeric Arbitration Preregistration

Date: 2026-07-01

## Objective

MC012 tests the next post-MC011 bridge question:

> If same-format numeric answers are not enough, can changing the
> source/evaluation contract create a clean local-versus-learned behavior
> mixture?

MC011 showed that prompt-local lab numbers and learned atomic numbers can share
the same integer answer interface while conflict rows remain entirely
prompt-local. MC012 keeps the numeric interface but makes source reliability the
experimental variable.

## Behavior Contract

The runner is:

> `code/mc012_reliability_labeled_numeric_arbitration.py`

The full table uses 40 chemical elements across source-disjoint discovery,
calibration, and holdout splits. It creates seven panels:

- `synthetic_numeric_lookup`;
- `familiar_entity_numeric_lookup`;
- `real_world_atomic_number_control`;
- `trusted_source_conflict`;
- `untrusted_source_conflict`;
- `neutral_conflict`;
- `answer_absent_null`.

The two primary conflict panels are:

- trusted local source: expected local lab number;
- untrusted local source: expected learned atomic number.

The true atomic number is not printed in conflict prompts. The source-status
text is visible by design, so any passing behavior table is a prompt-channel
diagnostic until a separate locality control removes or matches that channel.

## Promotion Rule

MC012 can be behavior-ready only if one selected template satisfies all of:

- synthetic numeric lookup at least 90 percent local-number answers;
- familiar entity numeric lookup at least 90 percent local-number answers;
- real-world atomic-number control at least 85 percent atomic-number answers;
- answer-absent null at least 90 percent `UNKNOWN`;
- trusted conflict at least 85 percent local-number answers;
- untrusted conflict at least 85 percent atomic-number answers;
- primary conflict parseability at least 90 percent;
- non-holdout conflict has at least 10 local and 10 atomic/lure rows;
- holdout conflict has at least 4 local and 4 atomic/lure rows;
- candidate and output margins are reported.

## Death Rule

The route is not signature-ready if the only successful contrast is produced by
visible source-status text. In that case the correct verdict is:

> behavior-ready diagnostic, prompt-channel visible, hidden-state work blocked.

## Containment Rule

If the behavior table passes with prompt-visible source labels, the allowed
claim is only that explicit source reliability can manufacture a clean mixed
behavior substrate. The forbidden claim is that this is an internal
knowledge-control surface.

## Exported Diagnostics

- `RELIABILITY_BEHAVIOR_CONTRAST_PASSED`;
- `RELIABILITY_PROMPT_CHANNEL_VISIBLE`;
- `MC012_BEHAVIOR_READY_PROMPT_VISIBLE_BRIDGE`.

## Forbidden Claims

- MC012 is a mechanism card.
- MC012 supports intervention or steering.
- MC012 found an internal knowledge-control surface.
- A hidden-state signature should be searched before a materially different
  prompt-channel locality control passes.
