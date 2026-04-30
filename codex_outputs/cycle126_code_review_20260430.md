**Verdict**
Do not accept a PASS from the current analysis code. The training cells look usable, so this is fixable with `--reanalyze`, but the current analysis does not faithfully implement the prereg.

**Blocking Findings**
1. Missing preregistered baselines: `delta_telemetry`, `delta_Shesha`, and true `combined_non_geometry` are not implemented, even though telemetry/Shesha fields are being extracted and stored. Current combined baseline is only alpha/alpha²/early-loss. See [analysis baselines](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:338>) vs prereg [baselines](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_186_dose_response_2026-04-29.md:97>).

2. It can PASS on fewer than the preregistered 48 rows. Rows with missing geometry are skipped, and analysis proceeds with `len(delta_X) >= 10`; criterion 1 requires pooled 48 delta rows. See [row skip](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:238>) and [too-few gate](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:260>).

3. If any geometry row is skipped, `delta_early_loss` becomes misaligned or crashes because it is built from full `labeled`, not filtered `delta_meta`. See [early loss baseline](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:338>).

4. Permutation test does not match prereg: code permutes labels, not geometry deltas, and runs 500 iterations, not 1000. See [permutation code](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:447>) vs prereg [shuffled geometry](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_186_dose_response_2026-04-29.md:104>).

5. Criterion 6 is under-enforced. Code allows PASS if only one architecture has a valid per-arch R²; prereg requires neither architecture negative. See [arch criterion](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:581>).

6. Preregistered fail condition “works only for alpha=1.0” is not enforced. Held-out-dose stress is reported only, not used in verdict. See [dose stress](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:557>) and prereg [FAIL criteria](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_186_dose_response_2026-04-29.md:167>).

**What Looks Correct**
The primary seed folds are not off by one: `(0,1)`, `(2,3)`, `(4,5)` are held out across both architectures, and standardization/Ridge fitting use train folds only. Implemented baselines are cross-validated correctly.

The bootstrap is now basically sound: it uses paired per-row geometry and baseline predictions, resampled by seed block. But it is only meaningful after the missing baselines and 48-row enforcement are fixed.

Training loop implements additive KD correctly: `loss = ce_c4 + kd_alpha * ce_t` at [line 132](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:132>). Feature extraction is at step 36 for 1200 steps, exactly 3%, after the 36th optimizer step: [feature step](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:66>).

**Performance**
Measured current board usage during the active run: about `20.9 / 24.5 GB`, with Python plus Ollama resident. WDDM did not expose per-process VRAM, so the experiment itself is probably lower, but margin is thin. Worst cell is Qwen3 KD because it holds C4 and teacher forward graphs/logits.

No obvious true GPU leak across cells, but cleanup happens before the function frame releases large locals, so PyTorch cache can stay high between cells. Safer cleanup is to empty cache in the caller after `train_one_cell_dose()` returns, or explicitly `del loss/logits/ids` before `empty_cache()`.

**Other Risks**
Feature extraction failure is swallowed and becomes `{}`, which can silently shrink the dataset. Gradient clipping uses `error_if_nonfinite=False`, so nonfinite grads could corrupt a cell despite finite loss. D4 scratch-denominator stability and normalized delta diagnostic are not implemented.

Bottom line: keep training, but patch analysis before interpreting. Existing cell records should be enough for a fixed `--reanalyze`; I do not see a reason to restart the 60-cell run.

