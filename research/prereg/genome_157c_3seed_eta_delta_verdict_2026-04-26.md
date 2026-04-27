# Pre-registration: genome_157c 3-SEED CANONICAL VERDICT eta/delta probe

**Date:** 2026-04-26
**Status:** LOCKED at first commit. CONDITIONAL: launches only if g157b PILOT returns DIRECTIONAL_SUPPORT_157b.
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Predecessor:** `research/prereg/genome_157b_eta_delta_probe_embedding_prefix_2026-04-26.md` (PILOT)

## 0. Why this prereg

g157b is a 1-seed PILOT. Per CLAUDE.md §4.1 and the locked PILOT prereg, PILOT data alone CANNOT promote claims from P-status (provisional) to C-status (confirmed) in CLAIM_EVIDENCE_MAP. The canonical verdict requires multi-seed replication.

g157c is the canonical verdict run: same protocol as g157b but with 3 training seeds (the full set already saved by g156: {42, 7, 13}).

## 1. Hypothesis (canonical, multi-seed)

For all 12 g156 checkpoints (3 seeds × 2 conditions × 2 arms), the layerwise transport surplus G_l = η̂_l − δ̂_l^mlp computed with embedding-layer prefix is:
- Positive in mid-band layers of natural-minimal (across ≥2/3 seeds)
- Non-positive in shuffled-minimal (mean over 3 seeds)
- Pooled contrast G_nat − G_shuf ≥ +0.03 nats on the minimal arm

## 2. System

Same as g157b PILOT, scaled to all 12 checkpoints:
- 12 checkpoints (full g156 set: {natural, token_shuffled} × {baseline_200M_4k, minimal_7L_200M_8k} × {seed 42, 7, 13})
- 3 mid-band depths per arm: 14L → indices [5, 7, 9]; 7L → indices [2, 3, 4]
- 2048 train / 256 val / 256 test windows from c4 VALIDATION split
- 13-token rolling-hash dedup audit (raises on >5%)
- FP32 probe weights + grad clip 1.0 + skip non-finite-loss
- 500 probe-train steps each (lin / local / prefix_embed)

## 3. Pre-stated criteria (CANONICAL — multi-seed)

- **PASS_C:** mean mid-band G_l on natural-minimal ≥ +0.02 in ≥2/3 seeds AND mean mid-band G_l on shuffled-minimal ≤ 0 (across all 3 seeds, pooled) AND pooled-seed contrast G_nat − G_shuf ≥ +0.03. Action: promote P12 → C16 in CLAIM_EVIDENCE_MAP. Update §0.1 score WIKI 6/10 → 7/10.

- **PARTIAL_C:** direction in 2/3 seeds OR contrast 0.015–0.03 nats. Action: keep as PARTIAL evidence; discuss whether to expand to 5 seeds or move to g158.

- **KILL_C:** in ≥2/3 seeds, natural-minimal G_l ≤ 0 OR shuffled-minimal G_l > 0. The PILOT was not replicable; the η > δ^mlp criterion is wrong. Treat as a false-pilot result.

## 4. Universality level claimed

If PASS_C: ⚪ → 🟡 (single-family transport-budget criterion measured at 200M Llama-3 scale). NOT yet Level-1; needs cross-class extension (g159).

## 5. Compute envelope

3× the g157b PILOT runtime ≈ 3 × 30 min ≈ 90 min. Within the 4-hr COMPUTE.md envelope.

## 6. What KILL_C means

If the PILOT directional-support fails to replicate at 3 seeds, the original signal was a single-seed artifact. Theory's η > δ^mlp criterion stays in the "unsupported" column. Falls back to Path C of `research/programs/post_g157b_decision_tree.md` (run g158 + g157 v3 control; if both KILL, mechanism dies, pivot to distillation).

## 7. Conditional launch

Launches ONLY if g157b returns DIRECTIONAL_SUPPORT_157b. If g157b returns WEAK_SUPPORT or KILL_157b, this prereg is archived (paths B or C take over).

## 8. Artifacts

- Reuse `code/genome_157b_eta_delta_probe_embedding_prefix.py` with `SEEDS = [42, 7, 13]` substituted for `PILOT_SEED = 42` (a 2-line change). Rename to `code/genome_157c_3seed_canonical_verdict.py`.
- `results/genome_157c_3seed_canonical_verdict.json`
- `results/genome_157c_run.log`
- Ledger entry per CLAUDE.md §4.6

## 9. Locking

LOCKED upon commit.
