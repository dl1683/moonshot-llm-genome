**Gate: APPROVE g191.** Run after g188 seed 13 direct-string cell lands, using g188 full-direct as the reference effect.

Use the exact g188 shell: GPT-2-tokenizer 8-layer Qwen3 config, C4 windows, 5000 steps, seeds `[42, 7, 13]`, `anchor_lambda=0.01`, same eval cadence. Do not use 28-layer Qwen3.

**Arms**
| Arm | Init | Anchor |
|---|---:|---:|
| `scratch_ce` | random | none |
| `direct_init_only` | full string-match matrix | none |
| `direct_anchor_only` | random | full string-match matrix |
| `matched_rows_only` | matched rows only | matched rows only |
| `unmatched_rows_only` | random | unmatched rows only |
| `row_shuffled_matched` | matched rows, permuted among matched tokens | same masked rows |
| `frequency_bucket_shuffle` | matched rows shuffled within GPT-2 train-frequency quintiles | same masked rows |

Implementation note: g188’s `direct_string_match_embeddings()` fills unmatched rows with matched-row mean. For g191, return `matched_mask` before fill. Mask anchor gradients rowwise for both `embed_tokens` and tied/actual `lm_head` params.

**Primary metric:** seed-matched `delta = scratch_nll - arm_nll` at step 5000.

**PASS Logic**
| Outcome | Interpretation |
|---|---|
| `matched_rows_only >= +0.35` mean, all seeds positive, and >=70% of full g188 direct effect | signal lives in exact matched token content |
| `unmatched_rows_only <= +0.05` mean | unmatched mean/fill rows are not carrying the effect |
| both shuffled arms `<= +0.10` mean and at least `0.25` below `matched_rows_only` | specific trained content, not row format/norm/spectrum/frequency |
| `row_shuffled_matched` high | format/spectrum/norm control explains effect |
| `frequency_bucket_shuffle` high but row-shuffle low | frequency structure is sufficient |
| `direct_init_only` high, `direct_anchor_only` low | warm start dominates |
| `direct_anchor_only` high | anchor itself is causal and more interesting for geometry-as-regularizer |
| both init-only and anchor-only weak, full/matched strong | init-anchor interaction is the mechanism |

**Wall-clock:** 21 cells × ~6.8 min = ~2.4h, plus ~5-15 min preprocessing/analysis. Total ~2.5-2.7h, inside the 4h / 22GB / 56GB envelope. Reusing g188 scratch would cut to ~2.1h, but rerun scratch if you want a clean standalone artifact.

**§0.1 movement:** strong content PASS moves this from 3.2/10 to ~4.0-4.3: real cross-tokenizer bridge, but bounded by exact shared strings. Anchor-only PASS adds more upside, ~4.4-4.7. Format/frequency PASS gives little or no uplift.

