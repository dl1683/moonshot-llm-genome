# e014b design — does training under a renormalized residual stream flatten the lesion map?

Status: DESIGN (implementable as `lab/e014b_stream_renorm.py`). Date: 2026-09-24.
Answers T003 Reading B (the authority-schedule hypothesis) and settles registered
prediction **P2**. Prior art: PARTIALLY-KNOWN (scratch/prior_art_authority_schedule.md)
— nGPT/post-LN build constant-norm streams but nobody has ever pointed a lesion
map at one. Our claim: the write/stream→damage link + the causal flattening test.

**Question.** In e001, attention damage falls monotonically L0→L5
([2.40, 1.74, 1.06, 0.38, 0.19, 0.03]) while the write/stream ratio falls 11.9×
(attn [4.06, 0.44, 0.54, 0.53, 0.46, 0.34]; stream norms at block inputs
[0.67, 5.66, 6.14, 5.12, 5.20, 5.62]). E011c showed damage tracks perturbation
ENERGY (geometry) more than content. If we pin the stream norm flat during
training — removing the geometric authority schedule by construction — does the
lesion map flatten (architecture owns the front-loading), or stay front-loaded
(learning owns it)?

**Core logic.** If the renormed net reaches parity (same loss) and its attention
lesion map is still front-loaded, then at parity every block's write/stream ratio
is `‖w_i‖/c` — the denominator can no longer explain depth structure — so the
front-loading must live in the learned write norms / task structure, not in the
stream-growth geometry. That is the registered falsifier.

---

## 1. ARMS

Common to all arms: `Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192,
block_size=256)`; `set_seed(42)` before build (identical init);
`CharCorpus(REPO/"data"/"input.txt", seed=1337)` — `train_model` seeds its batch
generator from `corpus.seed`, so **data order is bit-identical across arms**.
Training identical to e001: `steps=4000, lr=1e-3, batch_size=64`, AdamW
(wd 0.1, betas 0.9/0.95), grad-clip 1.0, cosine schedule (100-step warmup),
`max_seconds` per arm below, resumable ckpt via `train_model(ckpt=...)`
(saves model+opt+sched+gen_state+history at every eval point).

### (a) Baseline — exact e001 replica
Reuse `runs/checkpoints/e001.pt` (2,739,072 params, final val **1.6224**, ~2000
steps realized under the 240 s cap). Do NOT retrain (identical config/seed/data
by construction). Verify by recomputing val loss (expect 1.6224 ± 0.002; fp
nondeterminism tolerance) and the 12 block lesions (expect match to
`runs/e001/metrics.json` within ±0.02). If verification fails → retrain a fresh
arm from seed 42 at `max_seconds=240` (this is the only retrain trigger).

### (a′) CHEAP CONTROL (new, ~1 min): eval-only renorm of the trained baseline
Load e001.pt, install the renorm hooks (below) at eval only, measure val loss +
attention damage. Expectation under T003-B: a LARGE jump (order ≥ +0.3 nats,
plausibly several): renorm at block-0 input reweights the embedding 8.4×
relative to L0's writes (the trained mixture is off-manifold), even though each
individual LN is scale-invariant. Purpose: (i) establishes that renorm is not a
no-op on function (mixture ≠ scale); (ii) if instead Δ ≤ +0.05, the trained net
is already insensitive to stream mixture — an anomaly that itself weakens B and
must be flagged in THINKING before reading arm (b).

### (b) Renorm arm — THE intervention
After every block-boundary residual add — i.e., at every **block input** (6
sites, including block 0's input, which is the embedding stream `wte+wpe`) —
project the stream to a FIXED per-token norm **c = 5.6**:

```
x̃_i = x_i · c / max(‖x_i‖₂, ε),   ε = 1e-6,   per token (last dim, d=192)
```

Implemented as forward **pre-hooks on `model.h[0..5]`**, installed once after
build and NEVER removed — active in BOTH training and eval forwards (pre-hooks
fire regardless of `.train()/.eval()`; the arm's architecture includes the hook,
so lesioning/eval without it is off-manifold and forbidden).

**Why a constant, and why c = 5.6 (decision):**
- Per-depth targets (median baseline stream norm at that depth) are REJECTED:
  they re-encode the baseline's depth schedule — the object under test — into
  the intervention, and force an arbitrary choice at depth 0 (0.67 reproduces
  the 8.4× jump → null intervention; anything else → constant anyway).
- c = 1.0 (nGPT-style) is REJECTED: with writes at 1.8–5.6, a unit stream makes
  every block write 2–5× its stream — an INVERTED schedule, not a flat one.
- c = 5.6 = median of the baseline's post-embedding block-input norms
  [5.12, 5.20, 5.62, 5.66, 6.14] → within 9.6% of 5 of the 6 baseline block
  inputs. The intervention is therefore MINIMAL at depths 1–5 (downstream LN
  statistics see nearly their trained-typical scale) and MAXIMAL exactly where
  the hypothesis locates structural authority: the 0.67→5.66 embedding jump
  (block-0 input gets an 8.4× relative boost of embedding direction over L0's
  writes).
- One constant makes the logged write/stream ratios read directly as ‖w‖/5.6,
  comparable across arms and layers.

At parity with baseline write norms (attn [2.72, 2.49, 3.31, 2.73, 2.39, 1.93]),
the geometric authority term becomes [0.49, 0.44, 0.59, 0.49, 0.43, 0.34] —
dynamic range 1.7× instead of 11.9×. **The intervention removes ~85% of the
dynamic range of the geometric schedule, by construction, if writes stay put.**
(Whether writes stay put is itself an observable — §3.)

**Readout safety (why renorm cannot break the readout):** logits =
`lm_head(ln_f(x))` and LayerNorm is exactly scale-invariant —
`LN(αx) = LN(x)` for any per-token α > 0 (mean and std both scale by α). The
renorm is a per-token radial projection `x → α(x)·x`, α = c/‖x‖ > 0, so a
renorm placed at the `ln_f` input is a PROVABLE no-op (identical logits, the
projection is invisible to loss and gradients). Consequences: (i) the stream
entering `ln_f` (after block 5's adds) is left un-renormalized — pinning it
would change nothing and is omitted; (ii) renorm cannot damage the net through
the readout path — any loss gap in arm (b) comes from the changed input
mixtures inside the stack, not from a broken unembedding. Note the asymmetry
with the interior: inside the stack, renorm changes the MIXTURE of
(embedding-carried direction) vs (block writes) at each add — scaling one
summand of a sum is not a scale-invariance the LN can undo. That mixture change
IS the intervention. (Our renorm does not mean-center — it is a pure radial
pin, not a LayerNorm.)

**Gradient note:** the Jacobian of x·c/‖x‖ is (c/‖x‖)(I − x̂x̂ᵀ): it projects out
the radial direction. That is exactly the direction pre-LN already renders
invisible downstream (`ln1/ln2` are invariant to it), so the projection removes
no gradient information the baseline loss used. No warmup complications are
expected (this is NOT post-LN; sublayer inputs still pass through pre-LN ln1/ln2).

**Params:** unchanged, 2,739,072 (c is a constant, not learned) — capacity
parity with baseline.

### (c) Post-LN arm — robustness / fallback renormalized architecture
True post-LN blocks: `x = ln1(x + attn(x)); x = ln2(x + mlp(x))` — renorm with
mean-centering, learnable affines, at every sublayer add (12 sites). Implement
by subclassing, not hooks (hooks cannot reorder LN around the adds):

```python
class PostLNBlock(Block):
    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x

def to_post_ln(model):
    for i, blk in enumerate(model.h):
        p = PostLNBlock(model.cfg)
        p.ln1, p.attn, p.ln2, p.mlp = blk.ln1, blk.attn, blk.ln2, blk.mlp
        model.h[i] = p          # same param names → state_dict/ckpt compatible
```

Same param count, same seed/init. Train at lr 1e-3 first; post-LN's known
warmup sensitivity (Xiong 2020) at 6 layers is mild, but if train loss > 3.5 at
step 500 or NaNs, rerun (fresh, delete ckpt) at lr 6e-4 with warmup 300 —
requires a ~30-line local copy of `train_model` with a `warmup` parameter
passed to `cosine_lr`; acceptable inside the experiment file, fallback only.

**Run (c) when:** (i) always, as a robustness arm if time permits — if both (b)
and (c) flatten, the result cannot be read as "just post-LN"; (ii) MANDATORY as
PRIMARY if arm (b) fails the parity gate degenerately (§2). Priority order:
(a, a′, b, parity+lesions) → (c).

**Lesion-semantics note (applies to b and c):** zeroing attn/mlp at layer i in
a renormed arm removes its angular contribution while the next renorm re-pins
the norm — downstream LN statistics CANNOT shift (stream norm is pinned at c).
The renorm arms' lesion maps are therefore LESS confounded by the T001-H2
miscalibration artifact than the baseline's. Acknowledged asymmetry: lesions in
arm (b) measure pure angular damage; that is intrinsic to the intervention.

---

## 2. PARITY GATE (prerequisite for any lesion comparison)

- **Primary gate:** final val loss of arm (b) ≤ **1.7224** (= baseline 1.6224 + 0.10).
- **Secondary check (report both):** best val loss over eval points ≤ **1.65**
  (= baseline best ~1.55 + 0.10) — handles the shared overfitting-after-step-1500
  and matched-steps comparisons.
- Steps must match within ±5% of baseline's realized count (~2000); compensate
  the ≤5% hook overhead with `max_seconds=252` for arm (b) (vs 240 baseline).

**Fallback ladder (in order, each a resumable/fresh ckpt, each ≤ 30 min):**
1. Under-converged (val still descending at budget, gap > 0.10): extend
   `max_seconds` 252 → 600. Resume preserves optimizer/scheduler/data-order
   state exactly (`train_model` ckpt already implements this).
2. Unstable/diverged (train loss > 3.5 at step 500 or NaN): fresh retrain at
   lr 6e-4 (optionally warmup 300 via the local variant).
3. Still 0.10 < gap ≤ 0.30: report arm (b)'s lesion map as SECONDARY
   (confounded — flag the loss gap in every figure); promote arm (c) to primary.
4. Gap > 0.30 in BOTH (b) and (c): do not read P2 from lesions at all.

**What a large gap MEANS for T003 (pre-registered, so we can't retrofit):** if
renormed nets systematically cannot reach parity, the jump-then-plateau stream
is not an epiphenomenon of the lesion map — it is optimization-load-bearing
(the scheduler is woven into training itself; cf. TurnTrout: writes must scale
with the stream to sustain influence, and we forbade the stream from growing).
This SUPPORTS the depth of the authority-schedule claim while VOIDING the
clean lesion comparison. Note it would also qualify the naive nGPT reading:
nGPT changed the whole optimizer (normalized coordinate descent), we only
pinned the stream. Whatever the outcome, write it in THINKING before e014c.

---

## 3. INSTRUMENTATION (the second experiment hiding inside e014b)

**Training-path logging (self-counting hooks, no train_model changes).** The
renorm pre-hooks and attn/mlp forward hooks count their own invocations and
record, every 50th TRAIN-mode forward (tag by `module.training`), the mean
per-token norms: block-input norms ‖x_i‖ (must equal c — liveness assert) and
write norms ‖w_attn,i‖, ‖w_mlp,i‖. `.item()` sync only every 50 steps × 12
sites — negligible. Same probes on arm (a) (logging only, no renorm).

**Eval-path probe at every eval point** (and on final ckpt): one no-grad
forward over 4 fixed val batches (fixed generator seed) with temporary logging
hooks; collects block-input norms and write norms per layer. Cross-checks:
- arm (a) block-input norms must reproduce e011b's
  [0.67, 5.66, 6.14, 5.12, 5.20, 5.62] within 5% (validates instrumentation);
- arm (b) block-input norms must equal c within 1e-3 at EVERY eval point
  (validates hook liveness during training — a silent hook failure would make
  (b) a wasted baseline clone).

**The re-inflation observable (prior-art upgrade (c)).** With the denominator
pinned, the geometric authority term is ‖w_i‖/c — so if optimization WANTS an
authority schedule it can only rebuild it through the numerator:
- ratio trajectory: r_i(t) = ‖w_i(t)‖/c, every 50 steps;
- re-inflation factor (final, parity-gated):
  ρ_i = (‖w_i‖_renorm/c) / (‖w_i‖_base/‖x_i‖_base);
- **Re-inflation criterion:** mean_{i∈L1..L5} ρ_i ≥ 1.3, OR within-arm mean
  ratio rises ≥ +30% from step 250 to final.
- Interpretation ladder: re-inflation + flattening → schedule destroyed and not
  missed (cleanest B support). Re-inflation + NO flattening → optimization
  rebuilt the geometric schedule through write norms — the intervention failed
  to remove the geometric term and P2 is UNINFORMATIVE as a refutation, but the
  re-inflation is itself direct positive evidence that optimization constructs
  an authority schedule through whichever channel is open (headline finding;
  follow-up e014c: clamp write norms too). No re-inflation + no flattening →
  the strongest form of the falsifier (front-loading with a flat geometric
  term is pure learning/task structure).

**P3 within-arm check:** Spearman correlation of attention damage vs write/stream
ratio across the 6 layers, per arm (n = 6 — descriptive, not a test; T003-P3
registered ≥ 0.8 on the baseline).

---

## 4. LESION PROTOCOL + P2 OPERATIONALIZATION

Identical to e001 for every arm, hooks installed (arm-native forward):
- `lesion_loss(model, corpus, kind, layer, n_batches=30)` for attn/mlp blocks,
  24 for individual heads; deterministic `estimate_loss` (fixed generator
  seeded from corpus.seed → the SAME eval batches for all arms — paired
  comparison, batch-sampling noise cancels).
- **Primary comparison: attention block damage profile D′_0..D′_5 (arm b vs a).**

**Baseline reference (e001, re-verified):** D = [2.401, 1.737, 1.063, 0.379,
0.193, 0.034]; spread D_0−D_5 = 2.368; shares [0.414, 0.299, 0.183, 0.065,
0.033, 0.006].

**P2 (registered verbatim in T003: "flattens the attention damage profile by
≥50% (L5 damage rises well above +0.03; L0 falls below +2.0)"). Operationalized:**

- **P2 CONFIRMED** iff ALL of:
  1. spread′ = D′_0 − D′_5 ≤ **1.18** (= 50% of 2.368);
  2. D′_5 ≥ **+0.10** (≈3× baseline's +0.034, and ~5× the eval noise floor —
     see CIs below; this is "well above +0.03");
  3. D′_0 < **+2.00**.
- **P2 REFUTED (the falsifier)** iff parity gate PASSED and D′_0 ≥ +2.00 and
  D′_5 ≤ +0.05 (front-loading survives a flat geometric term → learning owns it).
- **Partial:** everything else — report spread reduction %, share profile
  (s_i = D′_i/ΣD′), and whether residual front-loading tracks the residual
  ratio profile (P3 within-arm).
- **Noise discipline (E011c caveat):** a local variant of `estimate_loss` that
  returns the 30 per-batch losses; bootstrap 2000 resamples → 95% CI per D′_i.
  Decision thresholds must have CI excluding the corresponding baseline value
  (in particular D′_5 ≥ 0.10 with CI lower bound > 0.034).

**Secondary registered predictions (register now, before running):**
- S1: arm (b) MLP-0 damage ≥ +2.0 (the +4.08 keystone is content — char
  statistics — not geometry; it should survive renorm). If MLP-0 also collapses
  (< +1.0), the front-loading was globally geometric after all.
- S2: re-inflation criterion of §3 fires in arm (b).
- S3: within-arm attention damage vs write/stream Spearman ≥ 0.8 in arm (a)
  (re-derivation of P3 on fresh numbers) and reported (not gated) for arm (b).
- MLP full profile logged both arms (E011a anomaly: late MLPs write large,
  dispensable content — does per-token renorm, which deletes the magnitude
  channel MLP-L5 was seen using as an ENERGY CARRIER (E011c), change MLP-L5's
  damage? Expect D′_mlp,5 to DROP if its value was mostly magnitude).

---

## 5. TIME / COMPUTE (lab rule: single steps ≤ 30 min, each arm resumable)

| step | wall clock | notes |
|---|---|---|
| (a) verify e001 ckpt + re-lesion 12 blocks | ~3 min | 30-batch evals, deterministic |
| (a′) eval-only renorm control | ~1 min | val + 6 attn lesions only |
| (b) train renorm arm | 240–252 s (≤ 4.5 min) | ~2000 steps realized, matches baseline; ckpt every 250 steps |
| (b) fallback extension (lr/warmup retries) | +8 to +16 min | only if parity fails |
| (b) lesions + bootstrap + probes | ~4 min | 12×30 + 36×24 batch evals |
| (c) post-LN arm (conditional) | ≤ 13 min | train (≤ 600 s cap) + lesions |

Renorm overhead: 6 pre-hooks/step, each a norm+mul over 64×256×192 floats
(~10 MFLOPs vs ~GFLOPs of block matmuls) + ~6×20 µs Python overhead → measured
expectation ≤ 5% wall clock (hence max_seconds 252, and verify matched final
step count ±5%). Total core run ≤ 15 min; absolute worst case with every
fallback ≈ 35 min spread over separate invocations, none > 30 min.
Outputs: `runs/e014b/` (metrics.json + one 3-panel figure: attention damage
profiles overlay; ratio profiles + re-inflation trajectories; training curves),
ckpts at `runs/checkpoints/e014b_{renorm,postln}{,.train}.pt`.
Single seed 42 for decide-ability (matches lab norm); the standing single-seed
critique (T001 amendment #10) is answered by follow-up replication seeds
(e014b.1: seeds 43/44 for arms a+b only) — not blocking.

---

## 6. FAILURE MODES → interpretation (pre-registered)

| outcome | meaning for T003-B |
|---|---|
| parity + P2 confirmed (flatten) | Architecture owned ≥ half the front-loading. B's strong form advances to "causally demonstrated at 6L/char-scale" (with TurnTrout's Pythia/GPT-Neo shrink counterexamples bounding universality). |
| parity + P2 refuted (sticky front-load, no re-inflation) | Strong-form refutation: with a flat geometric term, learning still front-loads → task structure (early aggregation at char level) owns the map. Matches Gromov/Nepal stickiness. P3 remains a correlation, causally carried by learned write norms. Clean negative, publishable. |
| parity + partial flatten | Quantify: % spread reduction; check residual front-loading vs residual ratio (P3 within-arm). Mixed verdict — geometry contributes a measured share. |
| re-inflation + no flatten | Intervention failed to remove the geometric term (it moved to the numerator); P2 uninformative as refutation, but re-inflation alone shows optimization WANTS an authority schedule — independent positive evidence for B. Queue e014c (clamp ‖w‖ as well, LayerScale-style). |
| small gap (0.10–0.30) after fallbacks | Confounded; arm (c) becomes primary; report arm (b) flagged. |
| large gap (> 0.30) in both (b) and (c) | Stream growth is optimization-load-bearing: the schedule is real architecture, so deeply built in that removing it costs loss. P2 unresolvable here; the gap is itself the finding (qualifies nGPT's speed claim: they changed the optimizer too). |
| divergence/NaN | treat as the large-gap row after lr/warmup retries. |
| (a′) renorm barely hurts trained baseline (≤ +0.05) | Anomaly: the trained net is insensitive to stream mixture — weakens B independently of arm (b); think before running (b) interpretation. |
| hook-liveness assert fires | Do not interpret anything; fix and rerun (a silently inactive hook makes (b) a baseline clone). |

Honesty reflex (README): the finding is behavioral (lesion deltas at parity),
the intervention is causal (training-time), and the mechanism is measured
(write/stream logs) — all three survive this design.

---

## 7. PSEUDOCODE — exact training-loop modification

```python
C = 5.6   # median of baseline post-embedding block-input stream norms

def install_stream_renorm(model: TinyGPT, c: float = C,
                          stats: dict | None = None, log_every: int = 50):
    """Per-token radial renorm to norm c at every block input.
    Active in train AND eval (forward pre-hooks fire in both modes).
    If `stats` is given, self-counts train-mode calls and records mean
    per-token block-input norms every log_every steps (liveness log)."""
    handles = []
    for i, block in enumerate(model.h):
        def pre(module, args, _i=i):
            (x,) = args
            n = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if stats is not None and module.training:
                stats["count"][_i] += 1
                if stats["count"][_i] % log_every == 0:
                    stats["block_in"][_i].append(float(n.mean().item()))
            return (x * (c / n),)          # differentiable; radial gradient
                                           # projected out (already invisible
                                           # to pre-LN downstream)
        handles.append(block.register_forward_pre_hook(pre))
    return handles

# ---- write-norm probes (eval-path, temporary) --------------------------
def probe_writes(model, corpus, n_batches=4):
    """One no-grad forward on fixed val batches; returns per-layer mean
    per-token ‖w_attn,i‖, ‖w_mlp,i‖ and block-input norms; removes hooks."""
    ...  # forward hooks on block.attn / block.mlp outputs + block pre-hooks,
         # collect t.norm(dim=-1).mean() per site

# ---- arm (b) runner -----------------------------------------------------
model, corpus = build()                          # set_seed(42); identical to e001
stats = {"count": [0]*6, "block_in": [[] for _ in range(6)]}
handles = install_stream_renorm(model, C, stats) # NEVER removed for this arm
history = train_model(model, corpus, steps=4000, lr=1e-3, batch_size=64,
                      max_seconds=252.0,
                      ckpt=REPO/"runs"/"checkpoints"/"e014b_renorm.train.pt")
# parity gate: history[-1]["val_loss"] <= 1.7224 (primary), min val <= 1.65
# liveness assert: |mean block-input norm - C| < 1e-3 at every probe
torch.save(model.state_dict(), REPO/"runs"/"checkpoints"/"e014b_renorm.pt")

# ---- lesion pass (hooks still installed!) -------------------------------
base_b = estimate_loss(model, corpus, "val", n_batches=30)
D_attn_b = [lesion_loss(model, corpus, "attn", i, n_batches=30) - base_b
            for i in range(6)]                   # + mlp + heads, as e001
```

Notes: (1) `model.state_dict()` is hook-independent — reloads must REINSTALL
hooks before any eval of arm (b) (spec it in the runner: `load → install →
eval`). (2) The `lesion()` context manager composes with the renorm pre-hooks:
zeroed writes at layer i flow into the pinned renorm at block i+1 — intended
semantics (§1 note). (3) `estimate_loss` calls `model.train()` at exit — hooks
tag by `module.training`, so eval probes stay clean. (4) Arm (a′): same
`install_stream_renorm` on the loaded e001 model, no `stats`, eval only.

---

## 8. Run order + bookkeeping

1. (a) verify → 2. (a′) control → 3. (b) train + gate + lesions → 4. (c) if
   mandated/possible → 5. bootstrap CIs, metrics.json (schema: per arm — val
   losses, gate booleans, D_attn/D_mlp/head profiles with CIs, write-norm and
   ratio profiles, ρ re-inflation factors, P2 booleans, S1–S3 booleans), PNG.
6. NOTES.md entry + THINKING T005 entry BEFORE anything builds on it (P2
   verdict + re-inflation verdict + parity verdict interpreted together, per
   the §6 table). Registered predictions above are frozen as of this file.
