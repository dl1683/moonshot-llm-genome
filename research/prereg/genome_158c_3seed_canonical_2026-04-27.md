# genome_158c — context-length inversion: 3-seed canonical verdict

**Status.** LOCKED 2026-04-27. Launched 2026-04-27 08:57 UTC.
**Code.** `code/genome_158c_3seed_canonical.py`
**Result.** `results/genome_158c_3seed_canonical.json`
**Forked from.** `code/genome_158_context_length_inversion.py` (PILOT, single seed=42).
**Decision tree.** `research/programs/post_g158c_decision_tree.md`.

## Hypothesis

The transport-demand input-side prediction (architecture-prior advantage is monotone in context length L) survives multi-seed canonical confirmation. Specifically, the g158 PILOT result (rho_c4 = +1.00, Delta_256 = +4.10pp, Delta_32 = -0.24pp at single seed=42) replicates across seeds [42, 7, 13] in mean.

## Setup

- Two arms at FLOP-matched per-cell budget (193.27 TFLOP/cell):
  - `baseline_6L+MLP` — 30M params, 6 layers, attention + MLP
  - `minimal_3L_noMLP` — 21M params, 3 layers, attention only (transport-only)
- Context lengths: L in {32, 64, 128, 256}
- Seeds: [42, 7, 13]
- LR selection: per-arm-per-context (best-min over LR sweep at L=32 and L=256, take min) at L=128 anchor
- Eval: C4 + Wikitext val (top-1 acc, NLL, OOD)

## Locked PASS / WEAK / FRAGILE criteria (decision tree)

**PASS_canonical:**
- mean rho across 3 seeds >= +0.8
- mean Delta_256(c4) 95% CI excludes zero AND mean >= +2.0pp
- mean Delta_32(c4) <= 0.0pp (negative or zero, never positive)

**WEAK_canonical:**
- mean rho >= +0.5 AND mean Delta_256 >= +1.0pp AND CI(Delta_256) does not flip sign

**PILOT_FRAGILE:**
- otherwise (rho < +0.5 OR CI crosses zero OR Delta_32 strongly positive)

## Compute envelope (COMPUTE.md §9 compliance)

- Wall-clock estimate: ~5.5h (envelope overrun documented; see WIKI cycle 21 framing)
- Peak VRAM: < 8 GB (one ~30M BF16 model live at a time, batch=8)
- Peak RAM: < 16 GB
- Disk: < 100 MB results JSON
- Quantization: BF16 forward, FP32 lr-selection probes (inherited from g158 PILOT design)

## Known limitations (Codex audit 2026-04-27)

- **Single C4 corpus across seeds.** All three seeds sample from the same `c4_clean_v1(seed=77)` pool. Run-seeds [42,7,13] vary model init + batch-order RNG only; corpus slice is shared. The "3-seed claim" therefore does not include data-slice variance. Documented and accepted: this matches the PILOT protocol.
- **No incremental checkpoint.** The script saves only at the end. A mid-run crash loses all work. Acceptable risk for a one-time canonical; future canonical re-runs must add per-seed checkpoint.
- **CUDA non-deterministic.** Same-seed reruns may drift slightly. Not a verdict-validity issue; documented.
- **lr_select_L metadata.** The JSON records `lr_select_L=128`, but the actual policy is "select at L=32 and L=256, take min". The eval metadata is wrong; the actual selection policy is correct (per Codex 158 pre-flight). Future re-runs must fix the metadata to match policy.

## Verdict integration

When the run completes:
1. Run `python code/integrate_g158c.py --commit` → emits ledger entry, WIKI patch, applies decision-tree adjudication.
2. Update CLAIM_EVIDENCE_MAP P13 → C17 (PASS) / P13 → WEAK_CANONICAL (WEAK) / P13 → R8b (FRAGILE).
3. WIKI patch lands in same commit.
4. Per Codex direction consult 2026-04-27, next experiment is path-conditional:
   - PASS_canonical → g162 transport-arm capacity sweep (Path A)
   - WEAK_canonical → g158e endpoint seed expansion (Path B; NOT g158d budget)
   - PILOT_FRAGILE → g155 production distill + locked C3 TEI/kJ (Path C; HARDWARE-BLOCKED on wall-power meter)

## Provenance

- PILOT result: `results/genome_158_context_length_inversion.json`, ledger entry `genome_158_context_length_inversion`
- PILOT prereg: `research/prereg/genome_158_PILOT_2026-04-27.md`
- Decision tree: `research/programs/post_g158c_decision_tree.md` LOCKED 2026-04-27
- Codex pre-flight audit: `codex_outputs/heartbeats/g158c_audit_20260427T090500.md`
- Codex direction consult: `codex_outputs/heartbeats/post_g158c_design_20260427T090500.md`
