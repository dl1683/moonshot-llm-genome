# Pre-registration: genome_157b eta/delta probe — EMBEDDING-LAYER PREFIX VARIANT

**Date:** 2026-04-26
**Status:** LOCKED at first commit. CONDITIONAL: launches only if g157 PILOT returns PILOT_KILL or WEAK_SUPPORT due to suspected probe-design issue (same-layer prefix probe captures already-transported info, leaving little for q_prefix to add over q_local).
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Predecessor:** `research/prereg/genome_157_eta_delta_probe_pilot_2026-04-26.md`
**Codex source:** g159 lesion design + first-principles derivation (both alluded to embedding-layer prefix as a valid alternative)

## 0. Why this prereg supersedes / alternates

The original g157 PILOT uses h_<t at the SAME layer as the K/V for the prefix probe. The locked prereg explicitly allowed "or their embeddings plus h" as the prefix source. Early g157 PILOT data shows G_l < 0 on natural-minimal arm — the prefix probe is WORSE than the local probe across all 3 mid-band layers. This is consistent with:

**Hypothesis (probe-design issue, not theory failure):** at layer ℓ, the same-layer h_<t already contains the prefix info that the model has transported by layer ℓ. The cross-attention probe given access to h_<t can only extract info that the residual stream already encodes — which q_local can also extract from h_t alone (if the residual stream merged it correctly). In the limit where attention has transported all useful prefix info into h_t, q_prefix and q_local should give identical CE.

The PROPER test of "remaining transport gap" uses h_<t from the **embedding layer (layer 0)** — fresh prefix tokens before any transport has occurred. Then q_prefix captures "how much MORE info the model could have transported but didn't."

## 1. Hypothesis

For the seed=42 subset of g156 checkpoints, the layerwise transport surplus G_l = η̂_l − δ̂_l^mlp computed with EMBEDDING-LAYER prefix is positive in mid-band layers of natural-minimal AND non-positive in shuffled-minimal.

## 2. System

Same as g157 PILOT (4 checkpoints, 3 mid-band depths, BF16, 500 probe steps). Only difference: the PrefixAttnProbe's K/V come from `model.model.embed_tokens(prefix_tokens)` (or the post-embedding layernorm output if the architecture uses one) — not from the same-layer activations.

## 3. Probes (same as g157 PILOT except q_prefix)

- **q_lin(y|h_t):** linear softmax probe on h_t at layer ℓ
- **q_local(y|h_t):** 2-layer token-local MLP probe on h_t at layer ℓ
- **q_prefix(y|h_t, embed(prefix)):** one-head causal cross-attention; query = h_t at layer ℓ, K/V = embed(prefix tokens)

The embedding layer is shared across both arms (same Pythia tokenizer + same trained embed table per checkpoint). Note: the embed table itself is trained, so this is not "raw prefix" — it's "prefix at the model's input representation."

## 4. Pre-stated criteria

Same as g157 PILOT:
- **DIRECTIONAL_SUPPORT:** mean mid-band G_l ≥ +0.02 nats on natural-minimal, ≤ 0 on shuffled-minimal, contrast ≥ +0.03 nats. Action: write 3-seed prereg.
- **WEAK_SUPPORT:** contrast 0.015–0.03. Action: redesign probe further.
- **KILL_157b:** even with embedding-layer prefix, no positive G_l on natural-minimal. Action: theory's η_l > δ_l^mlp criterion is wrong; transport-theory mechanism dies. Pivot to distillation track.

## 5. Compute envelope

Same as g157 PILOT (~30 min projected). Same hard-abort if microbenchmark > 3.5 hr.

## 6. What KILL_157b means

If both same-layer prefix (g157 PILOT) AND embedding-layer prefix (this g157b) fail to show G_l > 0 on natural-minimal, the η_l > δ_l^mlp criterion is empirically wrong. The transport theory still has g156 PASS (cross-axis falsifying inversion observed) but loses its proposed internal mechanism. We pivot the breakthrough-axis claim to "transport-vs-local-axis is real; the η/δ budget criterion is one possible mechanism but not validated."

## 7. Artifacts

- `code/genome_157b_eta_delta_probe_embedding_prefix.py` (NEW)
- `results/genome_157b_eta_delta_probe.json`
- `results/genome_157b_run.log`
- Ledger entry per CLAUDE.md §4.6

## 8. Locking

LOCKED upon commit.
