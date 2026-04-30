# g198 Prospective Canary Policy — DRAFT (NOT LOCKED)

**Status:** DRAFT. Gated on g197 PASS_CANARY. This prereg will be finalized and locked only after g197 results are in and the Ridge predictor is frozen.

**Purpose.** Freeze the g197 geometry→NLL predictor. Create 8-10 novel lm_head conditions NOT in g197's training set. Score each at step 0/50. Only continue training predicted-good conditions. Show compute saved or bad runs killed before 5000 steps. Moves the project from "interesting predictor" to "operational training triage."

**§0.1 impact:** +0.8 to +1.3 if successful (per Codex Architecture-Theorist cycle 198).

## Novel Conditions (out-of-distribution for g197)

| Condition | Construction | Expected regime |
|---|---|---|
| `pca_top32_trained` | project trained rows into top-32 PCA dims, zero-fill rest | wasteful |
| `sparse_top3_trained` | keep only top-3 magnitude elements per row | wasteful/healthy |
| `interpolated_0.5` | 0.5*trained + 0.5*random_gaussian | intermediate |
| `interpolated_0.25` | 0.25*trained + 0.75*random_gaussian | neutral/wasteful |
| `negated_trained` | negate all trained row directions | doomed |
| `rotated_15deg` | rotate trained rows by 15° around random axis | wasteful |
| `rotated_45deg` | rotate trained rows by 45° around random axis | doomed |
| `freq_reordered` | assign trained rows by frequency rank | wasteful |
| `quantized_4bit` | round trained rows to 4-bit representation | healthy/wasteful |
| `llama_trained` | use Llama-3.2-1B lm_head rows (different model family) | wasteful/doomed |

## Frozen Predictor

The Ridge model from g197 (trained on 10 conditions × 3 seeds = 30 cells) is frozen. Its feature weights and alpha are locked. It predicts NLL for each novel condition at step 0 and step 50.

## Pass Criteria (DRAFT)

**PASS_CANARY_POLICY:** The frozen predictor correctly ranks at least 7/10 novel conditions by final NLL (Spearman rho >= 0.7), AND the predicted-good conditions (top 3 by predicted NLL) actually outperform the predicted-bad conditions (bottom 3) by at least 0.15 nats average gap.

**FAIL:** Spearman rho < 0.4 OR predicted-good do NOT outperform predicted-bad.

## Compute Savings Metric

Report: if we had used the predictor to kill bottom-50% conditions at step 50, how many cell-hours would we have saved vs. running all 30 cells to completion?
