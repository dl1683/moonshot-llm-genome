# Pre-registration: genome_157 PILOT layerwise eta/delta probe (relocked)

**Date:** 2026-04-26 (RELOCKED after Codex pre-flight `codex_outputs/g157_pre_flight.md` flagged 91-hr compute estimate on the original 3-seed spec)
**Status:** LOCKED at first commit. SUPERSEDES `genome_157_eta_delta_probe_2026-04-26.md`.
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Program ref:** `research/programs/post_g156_pass_program.md` §g157
**Prerequisite:** g156 PASS_TRANSPORT verified (commit 06e5203, 12 saved checkpoints)

## 0. Why this prereg supersedes the original

The original 3-seed × 5-depth × 3-probe × 3000-step spec computed to ~91 GPU-hours per Codex pre-flight (full-vocab output × 540,000 probe-train steps). That is 22× over the 4-hr COMPUTE.md envelope.

This relock SHRINKS scope to a 1-seed PILOT to validate the primitive's mechanics first, then expands to multi-seed only after the pilot result is in. Conditional reasoning:
- If pilot returns directional support (G_nat > 0 mid-band, G_shuf ≤ 0), expand to 3 seeds (writing a NEW prereg) for the locked verdict.
- If pilot returns clear failure (G_nat ≤ 0 or non-collapsing), the theory's mechanism is wrong and we don't waste 30 GPU-hr re-verifying.

The 1-seed pilot is NOT a bypass of the 3-seed requirement; it is a feasibility check. The locked PASS verdict still requires a 3-seed run under a separate prereg.

## 1. Pilot hypothesis

For the **seed=42** subset of the g156 checkpoints (4 ckpts: natural+shuffled × baseline+minimal), the layerwise transport surplus G_l = η̂_l − δ̂_l^mlp is positive in mid-band layers of the natural-minimal arm AND non-positive in mid-band layers of the shuffled-minimal arm.

## 2. System (PILOT scope)

- **4 checkpoints only** (seed 42 subset of g156): `{natural, token_shuffled} × {baseline, minimal}`
- **3 mid-band depths only**: 14L → indices {5, 7, 9}; 7L → indices {2, 3, 4} (locked tables)
- Probe data: 2048 train / 256 val / 256 test windows from **c4 VALIDATION split** + 256 from Wikitext-103 VALIDATION split
- 13-token rolling-hash dedup audit between probe data and g156 train data; abort if >5% overlap
- Tokenizer: Pythia GPT-NeoX (matches g156)

## 3. Probes (BF16 throughout per envelope discipline)

For activations h_t^l at the chosen layer:
- **q_lin(y|h):** linear softmax probe (Linear → vocab)
- **q_local(y|h):** 2-layer token-local MLP probe with mlp_hidden tuned for ±0.5% param match to lin
- **q_prefix(y|h, prefix):** one-head causal cross-attention probe over same-layer prefix activations; kv_dim tuned for ±1% param match (kv_dim ≈ 965 for hidden=1024, vocab=50277)

All probes: BF16 weights and activations; logits cast to FP32 only inside cross_entropy.

Training: 500 steps, batch=32, AdamW lr=1e-3, weight_decay=0.01. Best-on-val checkpoint reload at end. Eval on test split.

## 4. Estimator

Per layer:
- `delta_hat^mlp(l) = CE_lin(l) − CE_local(l)`
- `eta_hat(l)       = CE_local(l) − CE_prefix(l)`
- `G_l              = eta_hat(l) − delta_hat^mlp(l)`
- `R_l              = eta_hat(l) / max(delta_hat^mlp(l), 1e-6)`

## 5. Pre-stated criteria (PILOT — not multi-seed PASS verdict)

- **DIRECTIONAL_SUPPORT:** mean mid-band G_l ≥ +0.02 nats on natural-minimal AND mid-band G_l ≤ 0 on shuffled-minimal AND contrast G_nat − G_shuf ≥ +0.03 nats.
  → Action: write a 3-seed locked prereg and execute for the multi-seed verdict.

- **WEAK_SUPPORT:** direction same but signal weaker (contrast 0.015–0.03 nats).
  → Action: redesign probe (more steps? bigger probe params? different mid-band?) before scaling.

- **PILOT_KILL:** G_nat ≤ 0 OR shuffled G ≥ natural G (no collapse).
  → Action: theory's mechanism is wrong. Stay with empirical g156 PASS as cross-axis evidence; pivot to distillation track (g154→g160).

## 6. Universality level claimed

**None.** Pilot only.

## 7. Compute envelope (COMPUTE.md §9)

- VRAM: load 1 ckpt (~500 MB BF16) + activations on probe data (~2 GB peak streamed) + tiny probe (BF16) ~50 MB. Peak <8 GB. ✓
- RAM: activations sharded to disk after each layer; <20 GB resident at any time. ✓
- Wall-clock estimate: 4 ckpts × 3 layers × (extract ~30s + 3 probes × 500 steps × 0.3s BF16 = ~7 min) ≈ ~1.5 hr. ✓
- Disk: activation cache <10 GB under `cache/g157_activations/` (gitignored).
- Quantization: BF16 throughout. ✓

**Hard preflight abort:** the script runs a 100-step probe-train microbenchmark BEFORE entering the main loop. If projected total wall-clock > 3.5 hr, the script raises and we relock again.

## 8. Implementation differences from killed v1 (line-item)

- Loader: `c4` `split="validation"` (NOT `c4_clean_v1(seed=99)` train slice)
- Wikitext: `wikitext-103-raw-v1` `split="validation"` (NOT train)
- 13-token rolling-hash dedup audit added; raises on >5% overlap
- `depth_to_layer_idx` hard-coded to {14L: [5,7,9], 7L: [2,3,4]} for mid-band
- Completeness guard: assert exactly 4 ckpts found, 1 seed per arm-condition
- BF16 throughout probe training and eval
- PrefixAttnProbe kv_dim formula gives ±1% param match
- Activation extraction moves to CPU per-batch (no GPU concat-all)
- Microbenchmark + hard-abort if projected runtime > 3.5 hr

## 9. What a null result means

PILOT_KILL = the η_l > δ_l^mlp budget criterion is NOT directly observed at the layers we probed. The transport theory's mechanism may be wrong; g156's PASS could still stand as an empirical inversion result without this internal-quantity validation.

DIRECTIONAL_SUPPORT triggers the locked 3-seed run.

## 10. Artifacts

- `code/genome_157_eta_delta_probe.py` (REWRITTEN per this prereg)
- `results/genome_157_eta_delta_probe.json`
- `results/genome_157_run.log`
- `cache/g157_activations/` (gitignored)
- Ledger entry per CLAUDE.md §4.6

## 11. Locking

LOCKED upon commit. The original `genome_157_eta_delta_probe_2026-04-26.md` is now ARCHIVED (its locked criteria still hold for any future multi-seed verdict run, but it is NOT the active spec for this PILOT).
