# Probe-design lessons learned (2026-04-26)

Distilled from the g157 sequence (v2 KILL → v3/b/c/d branches). These apply to any future linear/MLP/cross-attention probe over frozen model activations.

## 1. ALWAYS use FP32 probe weights + grad clip

**Symptom in g157 v2:** lin probe CE = 230-290 on token_shuffled distribution.

**Mechanism:** BF16 cross_entropy on out-of-distribution data → exploding gradients → poisoned probe weights → inf logits → softmax saturates with high confidence on wrong tokens.

**Fix template:**
```python
probe = probe.to("cuda").to(torch.float32)  # FP32 weights, NOT BF16
opt = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=0.01)
for step in range(PROBE_STEPS):
    h_b = h_train[idx].to("cuda").to(torch.float32)  # FP32 activations during probe train
    logits = probe(h_b)
    loss = F.cross_entropy(...)
    if not torch.isfinite(loss):
        continue  # Skip non-finite step instead of poisoning params
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)  # CRITICAL for OOD distributions
    opt.step()
# At end:
if best_state is None:
    raise RuntimeError("probe never produced finite val checkpoint")
probe.load_state_dict(best_state)
```

The `if best_state is None: raise` is critical — silent fall-through to random weights produces bogus verdicts.

## 2. Same-layer prefix probes are structurally weak

**Question:** when probing how much "remaining transport gap" exists at layer ℓ, where should q_prefix(y | h_t, prefix) get its prefix from?

- **Same-layer prefix (h_<t at layer ℓ):** captures only info already merged into h_t by the residual stream up to ℓ. The cross-attn probe's added value is bounded by what the residual stream has already exposed. Rarely beats q_local(y | h_t).

- **Embedding-layer prefix (h_<t at layer 0 = embed_tokens output):** captures fresh prefix info that the model can choose to either transport or not. The probe's added value reveals how much the model SHOULD have transported but didn't.

**Empirical finding (g157 v2 vs g157b PILOT, both single-seed):**
- Same-layer (v2): natural-minimal mid-band G = -3.31 (eta = -1.21)
- Embedding-layer (b): natural-minimal mid-band G = ~-2.25 (eta ~= -0.4)

Embedding-layer prefix is strictly less negative — closer to 0, directionally consistent with "fresh prefix gives more discriminative signal." But still negative at single-seed pilot scale.

## 3. Probe budget needs to scale with class count

For full Pythia vocab (~50k classes), a probe with hidden→vocab projection has ~50M params. Training 500 steps × 32 batch = 16k samples gives ~3000 effective updates per param-group → severely undertrained. Either:
- More steps (g157d uses 2000)
- Larger probe rank (kv_dim 1500 in g157d)
- Smaller vocab via subsampling (changes the metric — not preferred)

## 4. Verdict thresholds must be calibrated to noise floor

**Codex audit on g152:** at 200M with 3 seeds, per-seed gap std ~0.35pp → 3-seed SE_mean ~0.20pp. PASS thresholds within this band are at the noise floor.

**Apply the rule:** for any pilot reporting Δ < 2 × SE_mean of the per-seed std, declare the result PROVISIONAL and require expansion to 5 seeds before claiming PASS or KILL.

## 5. Out-of-distribution probe data needs handling

For shuffled data (out-of-distribution relative to natural training):
- The model's hidden states have a different statistical distribution
- Probes trained on natural-distribution activations may fail on shuffled
- ALWAYS train probes on the SAME distribution they're evaluated on

Cross-distribution probes (train on natural, eval on shuffled) are a different experiment with different interpretation.

## 6. Microbenchmark + hard-abort BEFORE main loop

Codex pre-flight on g157 v1 caught a 91-hour estimate before launch. The fix: 50-step microbenchmark → projected total runtime → hard-abort if > envelope.

```python
# Run 50-step probe-train microbenchmark on synthetic data
# Multiply by total cells to project runtime
# Abort if projected > 3.5 hr
```

## 7. Verdict should record numerical-pathology flags

If a probe's CE > log(vocab) (random baseline), flag as PROBE_PATHOLOGY in the JSON. Verdict logic should treat flagged cells as missing-data (raise RuntimeError on incomplete cell), not silently aggregate.

## 8. Prereg locks experimental design, NOT implementation hardening

Adding FP32 + grad clip + microbenchmark + dedup audit + non-finite-loss handling to a PILOT does NOT invalidate its prereg lock. The lock covers hypothesis / criteria / thresholds / system spec. Implementation defenses against numerical pathology are protocol-preserving.

When in doubt, document the patches in the commit message and proceed.

## How to apply going forward

For g157c, g157d, g158, g159, g160, g161 and any future probe / measurement experiment:
- FP32 throughout probe training and eval
- Grad clip 1.0
- Skip non-finite loss steps with counter
- best_state-or-raise at end of train_probe
- 50-step microbenchmark → hard-abort
- 13-token rolling-hash dedup audit
- Per-cell completeness guard before verdict
