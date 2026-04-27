1. Severity 8: silent benchmark drop in [genome_160_transport_guided_student.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_160_transport_guided_student.py:198>). `load_c3_validation()` turns dataset failures into `[]`, and [measure_capability](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_160_transport_guided_student.py:154>) then averages over whatever tasks remain. A PASS/PARTIAL can therefore be emitted on 1-2 tasks while looking like full `C3_macro`.

Exact change:
```python
# after building `out` in load_c3_validation()
required = ("hellaswag", "piqa", "winogrande")
missing = [t for t in required if len(out.get(t, [])) == 0]
if missing:
    raise RuntimeError(f"C3 validation incomplete; missing/empty tasks: {missing}")
```

2. Severity 8: invalid ratio guard in [genome_159_cross_class_lesion.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_159_cross_class_lesion.py:470>). `R_nat = d_t_nat / max(d_l_nat, 1e-6)` and same for shuffled converts zero/negative local-lesion effects into huge positive ratios. That can create a false PASS when the denominator is noise or directionally wrong.

Exact change:
```python
if d_l_nat <= 0 or d_l_shuf <= 0:
    R_nat = float("nan")
    R_shuf = float("nan")
else:
    R_nat = d_t_nat / d_l_nat
    R_shuf = d_t_shuf / d_l_shuf

rat_nat = [per_depth[d]["R_nat"] for d in DEPTHS if np.isfinite(per_depth[d]["R_nat"])]
rat_shuf = [per_depth[d]["R_shuf"] for d in DEPTHS if np.isfinite(per_depth[d]["R_shuf"])]
if len(rat_nat) != len(DEPTHS) or len(rat_shuf) != len(DEPTHS):
    raise RuntimeError("non-positive local lesion delta; abort verdict")
R_nat_med = float(np.median(rat_nat))
R_shuf_med = float(np.median(rat_shuf))
```

3. Severity 7: NaN-skipping still allows bogus probe verdicts in [genome_157_eta_delta_probe.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_157_eta_delta_probe.py:160>) and [genome_157b_eta_delta_probe_embedding_prefix.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_157b_eta_delta_probe_embedding_prefix.py:159>). `train_probe()` just `continue`s on non-finite loss; if `best_state` is never set, the function returns an unvalidated/random probe and the script still writes a verdict.

Exact change:
```python
skipped_nonfinite = 0
...
if not torch.isfinite(loss):
    skipped_nonfinite += 1
    continue
...
if best_state is None:
    raise RuntimeError(
        f"probe training never produced a finite validation checkpoint; "
        f"skipped_nonfinite={skipped_nonfinite}"
    )
```

Otherwise clean on syntax/static pass: all touched files `py_compile`. Residual low-severity issue: non-ASCII source text is still present in g156/g157/g157b/g158/g160, which matches the Windows/mojibake failure mode the repo policy is trying to avoid.