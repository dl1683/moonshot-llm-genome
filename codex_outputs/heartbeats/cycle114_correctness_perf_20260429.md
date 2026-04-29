**Findings**

1. **Severity 8: partial resume can silently drop saved cells**  
In [code/genome_182_triage_arena.py](/c:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:2424), `all_cells = []`. On a resumed `--max-cells` run, only skipped cells encountered before the new limit plus newly trained cells are written back. Existing later cells are omitted from the next `save_incremental`, corrupting the result file.

Exact change:

```python
existing_cells = []
if existing and "cells" in existing:
    existing_cells = list(existing["cells"])
    for c in existing_cells:
        completed_cells[c["cell_id"]] = c
```

Then replace:

```python
all_cells = []
```

with:

```python
all_cells = list(existing_cells)
```

And remove both skip-path appends:

```python
all_cells.append(completed_cells[cell_id])
```

at the stage-1 and stage-2 `SKIP` branches.

2. **Severity 7: teacher cache validation fix is incomplete in replay / g184 paths**  
Main line 2401 now validates cache length, but `shesha_augment_main` and `frozen_eval_main` still trust cached teacher JSON. A short cache silently reduces `seq_kd_full` training examples and changes the replay/eval distribution.

Exact helper change:

```python
def load_teacher_text_cache(path: Path, expected: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        texts = json.load(f)
    if not isinstance(texts, list) or len(texts) < expected or not all(isinstance(t, str) and t for t in texts[:expected]):
        raise RuntimeError(
            f"Teacher cache invalid at {path}: got {len(texts) if isinstance(texts, list) else type(texts).__name__}, "
            f"expected >= {expected} non-empty strings"
        )
    return texts
```

Use it at lines 2401, 1415, and 1680 with expected `96 if smoke else N_TRAIN_WINDOWS + 512` for main, `N_TRAIN_WINDOWS + 512` for Shesha/g184.

No additional >=7 NaN, OOM, or Windows Unicode issue found. UTF-8 file I/O is mostly correct; NaN serialization is guarded by `allow_nan=False` plus optional-feature conversion.

