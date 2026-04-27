# Pre-registration: genome_157 layerwise eta/delta probe on g156 checkpoints

**Date:** 2026-04-26
**Status:** LOCKED at first commit adding this file. Active (g156 returned PASS_TRANSPORT 2026-04-26).
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Program ref:** `research/programs/post_g156_pass_program.md` §g157
**Prerequisite:** g156 PASS_TRANSPORT verified (commit 06e5203, 12 saved checkpoints under `results/genome_156_checkpoints/`)

## 0. End-goal alignment (CLAUDE.md §0)

g157 builds the missing measurement primitive that makes the prefix-information transport theory operational: the layerwise transport surplus G_l := η̂_l − δ̂_l^mlp. Without this primitive, g156's PASS is a single killer-test result. With it, the theory acquires a measurable internal quantity that future experiments can probe directly. This is the primitive the η/δ probe sketch in the derivation doc requires.

## 1. Hypothesis

On the 12 g156 final checkpoints, mid-layer transport surplus G_l is positive on natural-condition checkpoints AND collapses to non-positive on shuffled-condition checkpoints. This directly tests the budget criterion η_l > δ_l^mlp that the transport theory says drives the architecture-prior win.

## 2. System

- 12 saved checkpoints from g156: `{natural, token_shuffled} × {baseline_200M_4k, minimal_7L_200M_8k} × {seed 42, 7, 13}` at `results/genome_156_checkpoints/*.pt`
- Probe at FUNCTIONAL DEPTHS d ∈ {0.20, 0.35, 0.50, 0.65, 0.80} of each backbone (rounded to nearest layer index per arm: baseline has 14 layers → indices {3, 5, 7, 9, 11}; minimal has 7 layers → indices {1, 2, 3, 4, 6})
- Probe data: 4096 train / 512 val / 512 test windows from `c4_val_dedup_v2` (held out from g156 training), plus 512 from `wikitext103_val_v2` for OOD confirmation
- Tokenizer: Pythia GPT-NeoX (matches g156)

## 3. Probes (three per layer, equal training budget)

For activations h_t^l at the chosen layer:
- **q_lin(y|h):** linear softmax probe (W * h + b, classification over vocab)
- **q_local(y|h):** 2-layer token-local MLP probe (Linear → GeLU → Linear)
- **q_prefix(y|h, prefix):** one-head cross-attention probe — h queries the prefix tokens' embeddings (taken at the same layer, same arm) over an attention window of length t

Equal-budget = same parameter count for q_lin, q_local, q_prefix (modulo the cross-attn probe needing slightly different parameterization; tune so all three fit within 1% of identical param count). Same optimizer, same step count, same train/val/test split per layer.

## 4. Estimator

Per layer, on held-out test windows:
- `CE_lin(l)`, `CE_local(l)`, `CE_prefix(l)`
- `δ̂_l^mlp = CE_lin(l) − CE_local(l)` (best gain from token-local nonlinear decode)
- `η̂_l = CE_local(l) − CE_prefix(l)` (remaining transport gap after layer)
- `G_l = η̂_l − δ̂_l^mlp` (transport surplus; positive ⇒ attention budget should win)
- `R_l = η̂_l / max(δ̂_l^mlp, ε)` with ε=1e-6 (relative measure)

## 5. Pre-stated criteria (per locked program §g157)

- **PASS:** in the minimal-natural arm, mean mid-band G_l ≥ +0.02 nats in ≥2/3 seeds AND minimal-shuffled mean G_l ≤ 0.00 AND pooled contrast G_nat − G_shuf ≥ +0.03 nats. The transport budget criterion is observed AND collapses on shuffled.
- **PARTIAL:** same direction, contrast only ≥ +0.015 nats.
- **KILL:** no positive natural surplus, OR shuffle leaves the surplus unchanged within 0.01 nats.

Mid-band = mean over depths {0.35, 0.50, 0.65}.

## 6. Universality level claimed

**None.** Diagnostic primitive on a single architecture family. Building the η/δ probe coordinate; not yet promoting it to atlas Level-1.

## 7. Compute envelope (COMPUTE.md §9)

- VRAM: load 1 checkpoint (~500MB BF16) + activations dump on probe data + tiny probe training. Peak <12 GB. ✓
- RAM: activation cache for 5120 windows × 256 seq × hidden (1024) × 5 layers × 12 ckpts ≈ <20 GB if streamed; cache in shards. ✓
- Wall-clock: 12 ckpts × 5 layers × 3 probes × ~60s training each + activation extraction ≈ 1.5–2 hr. ✓
- Disk: activation caches sharded under `cache/g157_activations/`, gitignored.
- Quantization: BF16 throughout. ✓

## 8. What a null result means

KILL = transport surplus is not positive in the natural-arm regime where g156's PASS was observed. That would imply g156 PASS is real but the proposed mechanism (η > δ^mlp criterion) is wrong. Theory would need a new mechanism candidate — likely the rate-distortion / Zipfian story (Candidate 3 in derivation doc).

PARTIAL = direction supported but signal weaker than expected. Probably means probe budget is too small or layer subsampling is missing the key band; redesign before treating as discriminative.

## 9. Artifacts

- `code/genome_157_eta_delta_probe.py`
- `results/genome_157_eta_delta_probe.json` per (ckpt × layer) with CE_lin / CE_local / CE_prefix / G_l / R_l
- `results/genome_157_run.log`
- `cache/g157_activations/` (gitignored)
- Ledger entry per CLAUDE.md §4.6

## 10. Locking

LOCKED upon commit. Modifying hypothesis/criteria invalidates the prereg.
