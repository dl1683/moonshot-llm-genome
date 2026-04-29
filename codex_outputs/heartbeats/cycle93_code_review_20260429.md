Not clean. I would stop/restart the teacher generation now.

**S8: right-padding is a real generation bug.** In `generate_teacher_texts()`, decoder-only batched generation with right padding means shorter prompts generate from pad/eos positions. This can degrade/contaminate `seq_kd_full` for both archs.

Exact change:
```python
tok.padding_side = "left"
model.config.pad_token_id = tok.pad_token_id
...
out = model.generate(..., pad_token_id=tok.pad_token_id)
```

**S8: silent NaN feature failures are possible.** `g180.extract_features()` catches failures into `NaN`; g182 then median-imputes or zero-imputes them. That can turn broken geometry extraction into a “valid” run.

Exact change after each `features = extract_features_for_cell(...)`:
```python
required = AGNOSTIC_FEATURE_NAMES + QWEN_REF_FEATURE_NAMES
bad = [k for k in required if not math.isfinite(float(features.get(k, float("nan"))))]
if bad:
    raise RuntimeError(f"non-finite features cell={cell_id}: {bad}")
```
Also change `json.dump(..., default=str)` to:
```python
json.dump(data, f, indent=2, default=str, allow_nan=False)
```

**S7: finite loss check is not enough.** BF16/anchor paths can still produce non-finite grads. Replace:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
```
with:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
```

OOM: stage 1 batch size 8 looks plausible on 24GB, but donor embed/lm_head FP32 is loaded before stage 1 and kept on GPU unnecessarily. Defer donor loading until stage 2.

`.detach()` audit: lines 581/587 are fixed; other numpy conversions are either under `no_grad`, masks, or go through `g180._to_numpy()` which detaches. No missing parameter `.detach()` found.

Encoding: g182 code is ASCII. Docs are UTF-8 and will mojibake under cp1252, but g182 output is numeric/ASCII unless NaNs leak. 
