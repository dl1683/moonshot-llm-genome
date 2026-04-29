Found one bug.

**SEV-7** [code/genome_182_triage_arena.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:2225): permutation p-value is wrong.

```python
p_value = float(np.mean([1 for n in null_r2s if n >= real_mean_r2]))
```

This returns `1.0` if any null beats real, and `nan` if none do. It should be:

```python
p_value = float(np.mean([n >= real_mean_r2 for n in null_r2s]))
```

or `sum(...)/len(null_r2s)`.

Everything else requested looks correct:

- Permutation grouping is correct: labels are permuted only within `(arch, arm)` groups at lines 2208-2215.
- Arm-demeaned LOAO subtracts `(arch, arm)` group means correctly at lines 2134-2145. Note the comment says `arm_mean`, but implementation is stricter: `arch+arm` mean.
- Pairwise delta test matches seeds within the same architecture: `scratch_by_seed` is scoped per `arch`, then KD cells match by exact `seed`.
- Curvature fix is present: [code/genome_180_forecast.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_180_forecast.py:594) includes `transformer.h.{last_idx}.`.