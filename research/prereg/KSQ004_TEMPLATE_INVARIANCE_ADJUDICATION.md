# KSQ004 Template-Invariance Adjudication

Status: predeclared second-wave template-boundary adjudication.

Runner:

> `code/ksq004_template_invariance_adjudication.py`

Full behavior result:

> `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_full_behavior.json`

Status card:

> `research/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION_STATUS.md`

## Templates

- `question_form`
- `neutral_sentence_form`
- `relation_key_form`

## Promotion Rule

Admit bridge behavior only if at least two predeclared templates pass
matched pairs, direct controls, conflict routing, side-answer leakage,
null/holdout, and candidate/output margin gates. At least one passing
template must be non-question-form.

## Kill Rule

Kill same-family answer-interface repair if `question_form` is the only
passing surface or if compact/neutral templates keep collapsing
expected-atomic rows to local answers.

## Forbidden Claims

- This adjudication is a mechanism card.
- `question_form` alone is a robust bridge.
- First-token numeric margins replace sequence candidate scoring.
- A behavior pass permits hidden-state, intervention, or mechanism claims.
