# MC001G Gemma 2 2B Sparse Feature Discovery Preregistration

Date: 2026-06-30

Status: preregistered for sparse-feature signature discovery.

## Scope

- Card ID: `MC001G`
- Model: `google/gemma-2-2b`
- Sparse artifact source: `google/gemma-scope-2b-pt-res` via `sae-lens`
- Stage: sparse-feature signature discovery, no intervention claim allowed
- Runner: `code/mc001_gemma_sparse_discovery.py`
- Prior activation-patch status: `research/cards/MC001G_GEMMA2_2B_ACTIVATION_PATCH_STATUS.md`
- Repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- Result directory: `results/cards/MC001G/`

## Rationale

MC001G now has a repaired behavior substrate and matched dense signatures, but
both causal gates failed:

- additive layer 14 steering did not move matched holdout behavior in the
  intended direction;
- final-token activation replacement moved labels, but the movement was not
  donor-label-specific and caused locality failures.

The next useful question is therefore narrower: do public Gemma Scope residual
SAE features expose a sparse internal signature on the same matched rows that is
cleaner than the dense signature and non-tautological baselines?

This is a signature gate only. It cannot produce a mechanism card without a
later preregistered feature-level intervention.

## Fixed Data

Use the same repaired and matched MC001G rows as the pre-hint-margin discovery
and activation-patching gates:

- clean items only;
- conditions: `wrong_disclaimed`, `wrong_unsure`;
- labels: `truth_following` versus `user_agreement_error`;
- match field: item-level `no_hint` correct-minus-wrong log-probability;
- bin width: `0.5`;
- discovery/holdout split: inherited from the existing clean-item split.

The expected selected set is 48 rows: 34 discovery and 14 holdout.

## Sparse Artifacts

Use residual-stream Gemma Scope 2B pretrained SAEs through the official
SAELens loader:

- release: `gemma-scope-2b-pt-res-canonical`;
- layer 14 SAE ID: `layer_14/width_16k/canonical`;
- layer 20 SAE ID: `layer_20/width_16k/canonical`.

These are the width-16k canonical residual SAEs selected by SAELens. Layer 14 is
the best prior simple dense direction layer; layer 20 is the best prior dense
logistic-probe layer.

Use final-token residual-stream activations from the corresponding transformer
layer output, i.e. `hidden_states[layer + 1]` from the Hugging Face
`Gemma2ForCausalLM` forward pass. Encode and decode with `SAE.encode` and
`SAE.decode`; direct hand-rolled NPZ encoding is not accepted as evidence unless
it matches SAELens activation counts and reconstruction sanity checks.

## Metrics

For each layer:

- report SAE activation sanity checks, including mean active feature count;
- compute univariate feature AUCs on discovery rows;
- orient features only by discovery AUC;
- rank eligible features by oriented discovery AUC;
- evaluate the rank-1 and top-k sparse signatures on holdout;
- compare against same-row option margin, no-hint margin, no-hint margin bin,
  correct-hint margin, condition, answer-letter, and combined nuisance baselines;
- run label-shuffle nulls that choose top discovery features under permuted
  discovery labels and evaluate them on the real holdout labels.

Eligibility:

- a feature must be active on at least two discovery rows;
- a feature must be non-constant on discovery rows.

## Promotion Rule

Promote only to a feature-level intervention preregistration, not to a mechanism
card, if:

- the rank-1 discovery-selected feature or a fixed top-k sparse signature
  generalizes on holdout above the matched dense direction baseline;
- the result is not matched by the label-shuffle null distribution;
- the feature signal is not explained by non-tautological margin, condition, or
  answer-letter baselines;
- SAE activation sanity checks do not show an obvious encoding mismatch.

Do not promote if:

- strong holdout features appear only through holdout mining;
- the label-shuffle null frequently finds comparable holdout scores;
- sparse features do not beat the dense matched signature;
- activation counts or reconstruction sanity checks indicate the SAE is being
  applied to the wrong activation surface.
