# Pre-registration: g194 Scalar/Direction Factorial

**Status:** LOCKED (locked 2026-04-30 cycle 162, Codex design gate passed with SEV-8 Fro fix applied)

## Motivation

g191 PASS_CONTENT proves that exact matched trained embedding rows carry the signal (+0.465 nats), and that row shuffling is massively harmful (-0.709). But adversarial A17 (SEV-10) flags an unresolved confound: the signal could be per-token scalar norms (easy to learn) rather than directional content (hard to learn). g193's compiler FAIL (cosine=0.194 despite MSE=0.000926) independently shows that byte-level features capture norms but not directions — but this does not tell us which component of the trained row actually matters for training benefit.

## Hypothesis

**H1 (Direction):** The training benefit from matched rows is primarily carried by directional content (unit vectors u_t = e_t / ||e_t||). Correct directions with wrong norms still help; correct norms with wrong directions do not.

**H2 (Scalar):** The training benefit is primarily carried by per-token scalar norms (r_t = ||e_t||). Correct norms with random directions still help; correct directions with uniform norms do not.

**H3 (Both):** Both components contribute meaningfully; neither alone accounts for >80% of the full effect.

## Design

Decompose each matched trained embedding row: e_t = r_t * u_t where r_t = ||e_t||_2 (scalar) and u_t = e_t / ||e_t||_2 (unit direction vector in R^1024).

### Arms (6 arms x 3 seeds = 18 cells)

| Arm | Init embedding | Anchor | Tests |
|-----|---------------|--------|-------|
| `scratch_ce` | Random (Qwen3 init) | None | Baseline |
| `full_match` | Matched trained rows at correct positions | Same, masked to matched rows | Reference (+0.465 expected, replicates g191) |
| `correct_dir_shuffled_norm` | u_t * r_{sigma(t)} for matched rows | Same | Directions correct, norms shuffled across matched tokens |
| `shuffled_dir_correct_norm` | u_{sigma(t)} * r_t for matched rows | Same | Directions shuffled, norms correct per token |
| `random_dir_correct_norm` | random_unit_vector * r_t for matched rows | Same | Random directions, correct norms |
| `correct_dir_uniform_norm` | u_t * mean(r) for matched rows | Same | Correct directions, uniform (mean) norm |

All arms: 8-layer Qwen3-arch with GPT-2 tokenizer, 5000 steps, same data/eval as g191. Anchor lambda=0.01, masked to matched rows only. Unmatched rows: zero-filled (same as g191 matched_rows_only). **Frobenius normalization:** all factorial arms normalized to matched_fro (Fro norm of matched rows in full_embed), NOT trained_fro (full Qwen vocab Fro). This ensures all arms have the same matched-row energy as the full_match reference. Per Codex design gate (SEV-8 fix).

### Permutation details

- sigma is a random permutation of matched token indices (seeded per arm, NOT per training seed — same permutation across seeds for clean comparison)
- Random unit vectors: sample from N(0,1)^1024, normalize to unit length
- All init tables Frobenius-normalized to match trained_fro before use (same as g191)

## Pass/Fail Criteria

**PASS_DIRECTION:** correct_dir_shuffled_norm mean gain >= 0.30 nats AND (shuffled_dir_correct_norm mean gain < 0.15 nats OR random_dir_correct_norm mean gain < 0.15 nats). Directions carry the signal.

**PASS_SCALAR:** shuffled_dir_correct_norm mean gain >= 0.30 nats AND correct_dir_uniform_norm mean gain < 0.15 nats. Scalars carry the signal.

**PASS_BOTH:** Both correct_dir_shuffled_norm >= 0.20 AND shuffled_dir_correct_norm >= 0.20 AND neither alone accounts for >80% of full_match gain.

**FAIL:** No arm besides full_match achieves >= 0.20 nats mean gain (decomposition doesn't cleanly separate).

## Universality Level

Level-3 (architecture-specific, within Qwen3 family). Contributes to Mystery 8 resolution.

## Compute Envelope

- Same as g191: 8-layer Qwen3-arch, GPT-2 tokenizer, ~5 GB VRAM per cell
- 18 cells x ~7 min = ~2.1 hours total
- Well within COMPUTE.md envelope

## What a null result means

If FAIL: the signal is holistic (norm + direction jointly), not separable — the "content" is the full vector, not a projection. This would mean trained embedding rows are irreducible units of interface information.
