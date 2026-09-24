# MC005 Associative Lookup Source-Edge Preregistration

Date: 2026-06-30

## Question

Can a small language model's in-context key/value lookup be controlled through a
specific internal source edge rather than only through output logits or prompt
format?

This is a narrower target than MC001-MC004. The behavior is not broad factual
truth, refusal, or update conflict. It is a local associative lookup:

> Given in-context `key: value` pairs and a repeated query key, does the model
> retrieve the value paired with that key, and does masking the source-value
> attention path change that retrieval?

## Model

Primary model:

- `Qwen/Qwen3-1.7B`

Reason:

- local cached small LLM;
- same Qwen family as earlier diagnostics, but this is not another MC001 dense
  residual sweep;
- Hugging Face attention internals support eager per-head/per-layer source-edge
  masks already used by the repo's Qwen attention-source audits.

Fallback for speed/debug only:

- `Qwen/Qwen3-0.6B`

Fallback results must be labelled as debug unless rerun on the primary model.

## Behavior

Prompt family:

```text
Reference pairs:
- river: velvet
- garden: copper
- market: silver
- planet: orange
- window: marble
- river:
```

The next-token behavior is scored by forced-choice logit margin:

```text
target value first token - distractor value first token
```

The target value is the value paired with the repeated query key. The distractor
value is another value from the same prompt.

Rows are clean if the target-minus-distractor margin is positive before any
intervention.

## Splits

Rows are assigned to discovery and holdout before intervention selection.

Discovery may be used to:

- choose an attention layer/head by target-value versus distractor-value source
  attention;
- choose a causal layer by target-value source masking effect.

Holdout may not be used for selection.

## Candidate Signature

At the final query token, measure attention from each layer/head to:

- the source token containing the target value;
- the source token containing the distractor value.

Candidate-level labels:

- target source-value candidate: positive;
- distractor source-value candidate: negative.

Primary signature score:

```text
attention_mass(final_query -> candidate_value_token)
```

Selection:

- select the layer/head with highest discovery AUC over candidate-level labels.

Signature passes only if:

- selected discovery AUC is high enough to justify testing;
- selected holdout AUC is at least 0.85;
- selected holdout AUC is at least shuffled-label selection-null p95 + 0.05;
- holdout subgroup AUCs by target/distractor source order are each at least
  0.75 when both labels exist.

## Intervention

Intervention type:

- eager-attention source-edge mask at the final query token.

Source positions:

- primary: target source-value token;
- distractor control: distractor source-value token;
- random/source control: another value token from the same prompt;
- wrong-layer control: target source-value token at a non-selected layer;
- full-path diagnostic: target or distractor source-value token masked at all
  layers and heads.

Layer/head selection:

- layer/head signature selection uses discovery attention only;
- layer-level causal selection uses discovery target-value mask effect only;
- holdout intervention metrics are reported for both the selected attention
  head and the selected causal layer.

Intervention passes only if the holdout target-value source mask:

- reduces target-minus-distractor margin in the predicted direction;
- reduces it more than distractor-value, random-value, and wrong-layer controls;
- causes nontrivial target-win loss without broad collapse;
- has a full-path diagnostic effect stronger than the full-path distractor mask.

## Nulls And Controls

Required:

- shuffled-label selection null for attention signature;
- source-order subgroup AUCs;
- distractor-value source mask;
- random same-prompt value source mask;
- wrong-layer target-value source mask;
- selected-head target-value mask;
- selected-layer all-heads target-value mask;
- full-path all-layer/all-head target and distractor source masks.

## Promotion Rule

MC005 can move to a mechanism card only if all three gates pass:

1. Behavior: enough clean target-retrieval rows exist in discovery and holdout.
2. Signature: source-value attention identifies target versus distractor on
   holdout beyond shuffled-selection nulls.
3. Intervention: target source-edge masking changes holdout behavior in the
   predicted direction more than the controls.

If only the full-path source mask works, the result is a broad path/control
surface, not a localized mechanism.

If attention predicts but masking fails, the result is diagnostic only.

If masking works but attention selection does not survive holdout/nulls, the
result is source-removal control only.
