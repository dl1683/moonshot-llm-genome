# Pre-registration: genome_158 context-length inversion sweep

**Date:** 2026-04-26
**Status:** LOCKED at first commit adding this file. CONDITIONAL: launches only if g156 returns PASS_TRANSPORT or PARTIAL_TRANSPORT.
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Program ref:** `research/programs/post_g156_pass_program.md` §g158

## 0. End-goal alignment (CLAUDE.md §0)

This is the sharpest unique prediction of the prefix-information transport theory. If the architecture-prior win is driven by transport demand, shrinking context shrinks transport demand → win shrinks → at very short context the win inverts. The test produces a control-variable (context length) that the theory predicts can be tuned to make the architectural advantage disappear or reverse on demand. A single-test theory (g156 alone) cannot do this; an inversion sweep does.

## 1. Hypothesis

The architecture advantage of MLP-free attention+residual over MLP-equipped is monotonically increasing in transport demand. As context length L shrinks, transport demand shrinks, the minimal-arm advantage Δ_L := top1_minimal − top1_baseline shrinks and eventually inverts (baseline wins at very short L).

## 2. System

- 30M-class Llama-3 family (matches g141 family, but with proper FLOP accounting)
- **baseline_6L+MLP:** 6L, hidden=384, ffn=1024, num_heads=6 (matches g141 baseline)
- **minimal_3L_noMLP:** 3L, hidden=384, no ffn (ZeroMLP), num_heads=6 (matches g141 minimal)
- Tokenizer: Pythia GPT-NeoX (matches g141 line)
- Stimulus banks: `c4_train_dedup_v2` for training, `c4_val_dedup_v2` for primary eval, `wikitext103_val_v2` for OOD eval (NOT train as in g141..g151 — this is the §3.5/A1 fix)
- Dedup: 13-token rolling-hash exact overlap audit
- **N_TRAIN per condition:** 32768 sequences at the longest L=256; same TOKEN budget at shorter L (so N_TRAIN_L = N_TRAIN_256 × 256 / L) to keep total token-FLOPs comparable across L
- Context lengths: L ∈ {32, 64, 128, 256}
- Seeds: {42, 7, 13} for both training and shuffled-batch sampling

## 3. Arm-specific LR selection

Per the audit-hard protocol: arm-specific LR chosen on a SEPARATE validation bank (held out from c4_val_dedup_v2), then FROZEN for the full sweep. Candidate grid for each arm: {2e-4, 3e-4, 4e-4} (narrowed from g151's 4-LR sweep based on the g151 best-LRs landing at 2e-4 / 3e-4). LR is selected at L=128 (mid-range) per arm to avoid context-length-coupled selection bias, then applied identically to all four L values within that arm.

## 4. FLOP matching

Per the program's audit-hard protocol, total train FLOPs match within ±2% across arms within a given L. Use a frozen analytic FLOP counter:

```
FLOPS_per_token_per_layer ≈ 6 * d² (attn projections + matmul) + 4 * d * m (MLP) + (L * d) (attn matmul O(L*d) per token)
```

Total FLOPs = N_TRAIN × seq_len × n_layers × FLOPS_per_token_per_layer. Adjust step count per arm to match within ±2%.

## 5. Metrics

For each L × arm × seed:
- C4-val top-1 accuracy
- Wikitext-103-val top-1 accuracy

Per-L:
- Δ_L,c4 := mean(top1_minimal,c4) − mean(top1_baseline,c4) across seeds
- Δ_L,ood := mean(top1_minimal,ood) − mean(top1_baseline,ood) across seeds

## 6. Pre-stated criteria

- **PASS_INVERSION:** Spearman ρ(L, Δ_L,c4) ≥ +0.8 AND Spearman ρ(L, Δ_L,ood) ≥ +0.8 AND Δ_32,c4 ≤ −0.2pp AND Δ_256,c4 ≥ +0.5pp AND sign pattern of Δ_L matches across both eval sets at every L. Theory's inversion prediction validated.

- **PARTIAL_INVERSION:** Monotone-increasing pattern with Δ_256 ≥ +0.3pp AND Spearman ρ ≥ +0.6 in both eval sets, but no clean sign flip at L=32. Direction supported, magnitude weaker than predicted.

- **KILL_INVERSION:** No monotone increase in Δ_L (Spearman ρ < +0.3 in either eval set), OR minimal wins at all L (no decay toward zero). Theory loses its inversion axis.

## 7. Universality level claimed

**None.** Mechanism diagnostic, single-architecture-family. PASS would promote the *theory* but not yet to a Level-1 atlas claim. Cross-class extension is g159's job.

## 8. Compute envelope (COMPUTE.md §9)

- VRAM: 30M BF16 + activations at batch=8, max L=256 ≈ 2 GB peak. ✓
- RAM: 4× tokenized C4 pools (1 per L) + 2× shuffled controls if needed ≈ 8 GB. ✓
- Wall-clock: 4 L × 2 arms × 3 seeds = 24 cells. Per-cell ~4 min on RTX 5090 → ~1.6 hr. ✓
- Disk: result JSON + tokenized cache ≈ 1 GB. ✓
- Quantization: BF16 throughout. ✓

Within envelope.

## 9. What a null result means

**KILL_INVERSION** would mean the architecture-prior win is NOT controlled by transport demand. The theory loses its sharpest unique prediction. C13/g156 still stand as empirical results; the *transport explanation* for them dies.

**PARTIAL_INVERSION** suggests transport explains some but not all of the architecture-prior; we'd revise the theory to include a second mechanism (probably the rate-distortion / Zipfian story from Candidate 3 in the derivation doc).

## 10. Artifacts

- `code/genome_158_context_length_inversion.py`
- `results/genome_158_context_length_inversion.json` with per-(L × arm × seed) metrics + Spearman ρ + verdict
- `results/genome_158_run.log`
- Tokenized + dedup-audited corpora at the four L values
- Ledger entry per CLAUDE.md §4.6

## 11. Conditional launch

This experiment launches ONLY if g156 returns PASS_TRANSPORT or PARTIAL_TRANSPORT. If g156 returns KILL_TRANSPORT, this prereg is archived (not run); the transport theory is dead and there's no inversion to test.

## 12. Locking

LOCKED upon commit. Modifying hypothesis/arms/criteria invalidates the prereg.
