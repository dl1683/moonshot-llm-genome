# Pre-registration: genome_161 RWKV training-time transport extension

**Date:** 2026-04-26
**Status:** LOCKED at first commit. CONDITIONAL: launches only if g159 returns PASS or PARTIAL.
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Program ref:** `research/programs/post_g156_pass_program.md` §g161

## 0. End-goal alignment

g159 tests transport-vs-local asymmetry on PRETRAINED non-transformer models via lesion. g161 tests the same prediction at TRAINING TIME on a non-transformer. If both PASS, the transport principle is a class-general training-time design rule — not a transformer artifact and not just a lesion artifact.

## 1. Hypothesis

In small RWKV (linear-recurrent), removing channel-mix (the local sublayer) and re-investing budget into more time-mix layers (the transport sublayer) helps natural-text top-1 accuracy but NOT shuffled-text top-1 (or hurts it). The natural-vs-shuffled contrast that g156 observed in transformers replicates in RWKV.

## 2. System

Custom small RWKV-4 family with exact-FLOP matching (±2%):
- **baseline_rwkv:** `12L hidden=512 + channel-mix` (full RWKV-4 architecture)
- **transport_heavy:** `18L hidden=512 no-channel-mix` (channel-mix replaced with identity; budget re-invested in more depth)
- Tokenizer: Pythia GPT-NeoX (matches g141..g158 line; RWKV-4 vocab from registry)
- Stimulus: `c4_train_dedup_v2`, N_TRAIN sized for FLOP match
- Eval: `c4_val_dedup_v2` (1024 windows), `wikitext103_val_v2` (512 windows), and shuffled controls of both
- Seeds: {42, 7, 13}
- LR + warmup chosen on a SEPARATE validation bank per arm (matches post-g156 audit-hard protocol)

## 3. Conditions

- **natural:** standard tokenized C4 + Wikitext val
- **token_shuffled:** per-row token permutation (shuffle_seed=42 frozen; matches g156 protocol)

## 4. Metrics

For each (arm × condition × seed):
- C4-val top-1 accuracy
- Wikitext-103-val top-1 accuracy

Compute:
- Δ_nat = top1_transport_heavy − top1_baseline (mean over seeds, on natural)
- Δ_shuf = same on shuffled
- Contrast C := Δ_nat − Δ_shuf

## 5. Pre-stated criteria

- **PASS:** Δ_nat ≥ +0.3pp AND Δ_shuf ≤ +0.1pp AND C ≥ +0.3pp (on both eval sets).
- **PARTIAL:** Δ_nat ≥ +0.2pp AND C ≥ +0.2pp (on both eval sets).
- **KILL:** no contrast OR reversed contrast.

## 6. Universality level claimed

If g159 already PASS at Level-2 candidate, g161 PASS would tighten the case but the level-cap stays at Level-2 per §0.05 (no biology track in this moonshot).

## 7. Compute envelope (COMPUTE.md §9)

- VRAM: small RWKV-4 (~50M params) BF16 + activations at batch=8, seq=256 ≈ 4 GB peak. ✓
- RAM: tokenized data + shuffled control < 6 GB. ✓
- Wall-clock: 2 arms × 2 conditions × 3 seeds = 12 cells. Per cell ~10 min training (FLOP-matched to 200M-equivalent compute scaled down) → ~2 hr total. ✓
- Disk: result JSON, checkpoints gitignored. ✓
- Quantization: BF16. ✓

## 8. Conditional launch

Launches if g159 PASS or PARTIAL. If g159 KILL, theory does not generalize across classes; g161 is moot.

## 9. Audit-hard protocol

- 13-token rolling-hash dedup
- Wikitext-103 VAL split
- Exact analytic FLOP count match within ±2% across arms
- Arm-specific LR chosen on a SEPARATE validation bank then frozen
- Seeds {42, 7, 13}

## 10. What a null result means

KILL = transport-vs-local asymmetry that worked on Llama (g156) and lesioned RWKV/Falcon-H1 (g159) does NOT replicate when *trained from scratch* in a non-transformer architecture. Theory's training-time generality fails. Possible explanation: the lesion test in g159 detected a transport-pathway dependency that is real but architecture-specific; training a different architecture from scratch lets it route around any such dependency.

In that case, the claim line drops from "design law" to "diagnostic that detects which architectures already use prefix-information transport heavily." Still useful, smaller scope.

## 11. Artifacts

- `code/genome_161_rwkv_training_extension.py`
- `results/genome_161_rwkv_training_extension.json`
- `results/genome_161_run.log`
- Ledger entry per CLAUDE.md §4.6

## 12. Locking

LOCKED upon commit.
