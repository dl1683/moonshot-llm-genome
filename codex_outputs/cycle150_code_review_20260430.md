Found 3 severity ≥7 issues.

1. **S9 anchor strength bug:** dummy-model reference bug is fixed: targets are resolved by parameter name on the live training model. But `F.mse_loss` at `code/genome_188_tokenizer_flow_bridge.py:411-415` uses mean reduction, while g165/g181/g183 use Frobenius-sum anchor gradients. This makes the g188 anchor roughly `1 / numel` strength.

Exact change: remove the pre-backward anchor-loss block, then after `loss.backward()` add:

```python
if actual_anchor_pairs and anchor_lambda > 0.0:
    with torch.no_grad():
        coeff = 2.0 * anchor_lambda
        for param, target in actual_anchor_pairs:
            if param.grad is not None:
                param.grad.add_(param.detach().to(target.dtype) - target, alpha=coeff)
```

2. **S8 Sinkhorn reverses alignment strength:** `np.exp(-vals / ...)` treats character overlap as cost, so bigger overlaps get smaller weights. Exact change:

```python
scale = float(vals.max())
if not np.isfinite(scale) or scale <= 0.0:
    raise RuntimeError("invalid alignment weights")
vals = np.maximum(vals / scale, 1e-20)
```

3. **S7 λ mismatch:** as locked, g188 prereg says `ANCHOR_LAMBDA=0.0323`; code has `0.01`. If honoring the locked prereg, change line 58 to:

```python
ANCHOR_LAMBDA = 0.0323
```

If the intended protocol is “same as g183,” then code `0.01` is defensible, but the prereg must be re-locked/amended before interpreting results. Current code+prereg are not clean.

No SEV≥7 OOM or Windows encoding issue found. Result file currently has preprocessing only, no completed cells, so restart cost is low.

