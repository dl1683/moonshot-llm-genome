# g156 Audit Report

No critical implementation bug showed up in the shuffle/loss path. The core mechanics are correct: `shuffle_token_rows` at [g156:94](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:94>) gives a different permutation per row, preserves each row’s valid-token multiset exactly, and leaves pads fixed; the loss at [g156:149](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:149>) / [g156:110](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:110>) is computed against the next token of the shuffled sequence, with symmetric masking in train and eval; and `ZeroMLP` at [g156:67](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:67>) / [g156:88](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:88>) really does bypass FFN matmuls.

## Ranked Findings

1. **Severity 9** — The shuffled condition is still a learnable causal sequence, not an order-free control. `shuffle_token_rows` is one fixed per-row permutation ([g156:94], lines 94-107), then the model trains normally on those frozen sequences ([g156:218](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:218>), lines 218-228). RoPE and the causal mask still give the model position and prefix structure in the *new* shuffled order. That means a `KILL_TRANSPORT` result would not cleanly imply “natural prefix transport theory is false”; it could also mean “the architecture win survives on a different stable sequential signal.”  
   **Control:** if g156 is surprising, rerun `token_shuffled` with a fresh permutation every presentation, and average eval over several fresh shuffles. If the gap still survives there, the theory is in much worse shape.

2. **Severity 8** — There is a real robustness bug: failed seeds are silently dropped, but the script still emits PASS/KILL. `nan_seen` is recorded at [g156:247](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:247>), summary means ignore missing seeds at [g156:273](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:273>) lines 273-284, and verdict logic still runs at [g156:286](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:286>) lines 286-311.  
   **Exact fix:** insert this immediately before `delta_nat = ...`:
   ```python
   required_n = len(SEEDS)
   incomplete = [
       (cond_name, arm_name, summary[cond_name][arm_name]["n"])
       for cond_name in summary
       for arm_name in summary[cond_name]
       if summary[cond_name][arm_name]["n"] != required_n
   ]
   if incomplete:
       raise RuntimeError(f"Incomplete g156 run; missing valid seeds: {incomplete}")
   ```
   Without this, one bad shuffled seed can move `Δ_shuf` and still produce a formal verdict.

3. **Severity 8** — The shuffled condition reuses natural-tuned LRs and step budgets unchanged. `ARM_LR` is fixed at [g156:61](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:61>) and applied identically across both conditions at [g156:242](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:242>) / [g156:246](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:246>). If shuffled text changes optimization geometry, `Δ_shuf` can move because one arm is mistuned on the new task, not because transport vanished.  
   **Control:** run a shuffled-only LR spot-check on the g151 grid `{2e-4, 3e-4, 4e-4, 6e-4}` before treating `Δ_shuf` as mechanistic. One seed is enough for a sanity screen.

4. **Severity 7** — Three seeds are thinner at 200M than the 30M precedent suggests. `SEEDS=[42,7,13]` is fixed at [g156:53](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:53>). In `results/genome_147_matched_flops_200m.json`, the per-seed C4 gap std is about `0.35pp`; with `n=3`, the mean-gap SE is already about `0.20pp` before comparing two conditions. So the prereg kill band `|Δ_nat-Δ_shuf| <= 0.2pp` is close to the noise floor.  
   **Control:** if `|C| < 0.3pp` or either Δ lands within `0.2pp` of a prereg threshold, do not call PASS/KILL final; extend to 5 seeds. As of **April 26, 2026**, `results/genome_152_long_horizon_crossover.json` is not present in the workspace, so g147 is the only direct 200M multiseed variance estimate available.

5. **Severity 6** — Reproducibility drift versus prereg: the shuffled corpus is built on the fly at [g156:217](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:217>) lines 217-219 but never saved. That does not bias natural vs shuffled, but it violates the artifact plan and makes exact reruns depend on the live HF stream/tokenizer state.

6. **Severity 5** — Eval uses `SHUFFLE_SEED + 1` at [g156:219](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:219>) instead of the locked `SHUFFLE_SEED` at [g156:59](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:59>). This is not broken; distributionally it is still a random-permutation eval set. But it is a small prereg drift and an avoidable nuisance if the result is close.

7. **Severity 4** — The shuffle implementation is correct, but easy to misread. A single RNG is created at [g156:98](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:98>) and advanced per row at [g156:105](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:105>), so same-length rows do **not** share a permutation. Different rows get different fixed permutations; the same row gets the same permutation across reruns with the same shuffle seed.

8. **Severity 3** — Warmup has a small inherited off-by-one. `warmup_lr` at [g156:143](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:143>) combined with the 1-based loop at [g156:160](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_156_prefix_destruction_200m.py:160>) starts at `2/200 * lr` and reaches full LR one update early. Because g151 used the same code path, this is not a natural-vs-shuffled asymmetry.

## Single Biggest Risk

The biggest risk is **misinterpreting `KILL_TRANSPORT` as a clean theory falsification when `token_shuffled` still preserves a stable sequential language**. The implementation destroys *original natural order*, but it does **not** remove causal-prefix modeling, positional structure, or document-level bag/position correlations under a fixed one-time shuffle. If g156 returns KILL, I would treat that as “the current transport story is not sufficient under this control,” not as “ordered-prefix transport is definitely irrelevant,” unless the fresh-reshuffle control also kills the gap.

## Pre-Flight Checklist

1. Add the incomplete-seed guard above; if any arm/condition finishes with `n < 3`, do not emit a verdict.
2. Save `train_ids_shuf`, `eval_ids_shuf`, `train_mask`, and `eval_mask` to the prereg cache artifact before training.
3. Decide whether eval should stay at seed `43` or be aligned to `42`; record the rationale before launch.
4. Run a 100-row shuffle audit: per-row multiset preserved, pads unchanged, same-length rows getting different permutations.
5. Run a token-frequency equality check on train and eval: natural and shuffled must match exactly within each split.
6. Verify `results/genome_152_long_horizon_crossover.json` exists after g152 finishes; if its 4k/8k seed-gap variance is high, promote g156 from 3 seeds to 5 before treating it as decisive.
7. Pre-commit an interpretation rule: if `|C| < 0.3pp` or any Δ is within `0.2pp` of a threshold, trigger extra seeds and a shuffled-only LR spot-check.
8. Queue the post-result control now: if g156 yields KILL or near-KILL, rerun shuffled with fresh per-presentation reshuffles before declaring the theory dead.

If you want the short version: **no critical shuffle/loss bug; the code is basically sound. The main danger is over-interpreting the shuffled condition, plus the silent `n<3` verdict bug.**