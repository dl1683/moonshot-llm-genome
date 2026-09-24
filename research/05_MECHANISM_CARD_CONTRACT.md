# Mechanism Card Contract

Every future claim should fit this format. If it does not fit, the project is not ready to claim it.

## Header

- Card ID:
- Date:
- Model:
- Model access:
- Target behavior:
- Practical motivation:
- Experiment owner:
- Status: preregistered, running, success, failed, inconclusive, retired

## 1. Behavior

Define the behavior in observable terms.

Required:

- positive examples;
- negative examples;
- ambiguous examples;
- label source;
- output metric;
- prompt families;
- holdout split;
- known confounds.

Question:

> What would convince a skeptical reviewer that the behavior is real and measured cleanly?

## 2. Candidate Signature

Define the internal pattern.

Required:

- layer and token position;
- instrument type;
- extraction method;
- feature/probe/vector/graph identifier;
- training data for the signature;
- heldout performance;
- alternative signatures considered;
- why this signature is not just an output artifact.

Allowed instruments:

- probe;
- activation direction;
- SAE feature;
- transcoder feature;
- attribution path;
- patching result;
- natural-language decoder hypothesis;
- parameter-localization result.

## 3. Intervention

Define the manipulation.

Required:

- intervention type;
- target layer/position/feature/parameter;
- strength or dose schedule;
- when applied;
- predicted direction;
- expected side effects;
- reversibility plan if applicable.

## 4. Nulls And Controls

Required:

- random direction or feature with matched norm;
- unrelated feature with matched activation frequency;
- wrong-layer intervention;
- wrong-token intervention;
- prompt-only baseline;
- output-only baseline;
- label permutation or shuffled control for signature discovery;
- off-target task set;
- fluency or coherence metric;
- adversarial prompt split.

## 5. Results

Required:

- behavior effect size;
- uncertainty;
- dose-response;
- side-effect table;
- holdout performance;
- examples;
- failures;
- raw artifact locations;
- analysis limitations.

Do not hide negative examples. A mechanism card without failures is not credible.

## 6. Reliability Atlas

Map behavior across:

- paraphrases;
- topics;
- languages;
- context lengths;
- model temperatures;
- model families;
- base versus instruction-tuned variants;
- adversarial prompts;
- multi-turn settings.

Use labels:

- works;
- weak;
- breaks;
- harmful side effect;
- untested.

## 7. Practical Implication

Answer:

- What engineering decision changes if this card is true?
- Does it beat output-only monitoring?
- Does it beat simple prompting?
- Does it reduce risk or improve capability with acceptable side effects?
- Is it worth more compute?

If the practical implication is unclear, mark it as scientific only.

## 8. Verdict

Allowed verdicts:

- **Mechanism supported:** signature, intervention, and reliability gates pass within scope.
- **Diagnostic only:** signature predicts behavior but interventions fail or are too harmful.
- **Control only:** intervention works but mechanism explanation is weak.
- **Artifact:** controls explain the result.
- **Inconclusive:** evidence is too weak or mixed.
- **Retired:** not worth further work.

## 9. Reviewer Checklist

- Did the card preregister its behavior and controls?
- Was the signature selected without leaking holdout data?
- Are nulls strong enough?
- Are off-target effects measured?
- Is the claimed scope narrower than the evidence?
- Does the intervention prove more than prompt engineering?
- Is there a failure section?
- Can someone rerun or audit the result?

## Claim Discipline

Never write:

- "This proves the model thinks..."
- "This is a universal mechanism..."
- "This feature is the concept..."
- "This edit fixes the behavior..."

Prefer:

- "Within this model and prompt family, this signature predicted the behavior and this intervention changed it under these controls."
