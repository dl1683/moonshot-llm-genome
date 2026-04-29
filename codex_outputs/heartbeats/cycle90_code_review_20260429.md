**Finding**

Sev-8 in [code/genome_180b_cross_tokenizer.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_180b_cross_tokenizer.py:1016>): `final_nll` is written without a finite check, and [cell_done](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_180b_cross_tokenizer.py:1161>) only checks key presence. Since `to_jsonable()` serializes NaN/Inf floats as `null`, a bad cell with a feature cache can resume-skip as complete, then either crash or poison/interrupt analysis at [build_completed_rows](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_180b_cross_tokenizer.py:1232>).

Exact change:

```python
def finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
```

After final eval:

```python
final_nll = finite_float_or_none(final_metrics.get("nll"))
if final_nll is None:
    raise RuntimeError(
        f"non-finite final NLL tokenizer={tokenizer_spec.label} "
        f"arm={arm.label} seed={seed}: {final_metrics.get('nll')}"
    )
```

Then store:

```python
"final_nll": final_nll,
```

And replace resume/row checks:

```python
final_nll = finite_float_or_none(result.get("final_nll"))
if final_nll is None:
    return False
```

```python
scratch_nll = finite_float_or_none(scratch.get("final_nll"))
if scratch_nll is None:
    continue
...
final_nll = finite_float_or_none(result.get("final_nll"))
if final_nll is None:
    continue
```

**Clean Checks**

`genome_182_triage_arena.py` already has the needed final-NLL guard at line 743. C4 validation seed split is present. Source files are ASCII. `python -m py_compile` passed for `genome_180_forecast.py`, `genome_180b_cross_tokenizer.py`, and `genome_182_triage_arena.py`.

Live PID `48644` is still running; result JSON read cleanly at review time with `21` cells, `0` bad/missing final NLLs.

