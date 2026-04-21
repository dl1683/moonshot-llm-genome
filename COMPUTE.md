# COMPUTE.md — Hardware Envelope and Constraint Rules

**This is the binding compute-constraint doc. Every design decision, every Codex prompt, every experiment plan must comply with the envelope defined here. No suggestion — from Claude, Codex, or any tool — may exceed this envelope without explicit user approval and a corresponding update to this file.**

Read first. Plan accordingly. Out-of-envelope proposals are rejected at the design gate.

---

## 1. Current hardware (verified 2026-04-20)

| Component | Specification |
|---|---|
| **GPU** | NVIDIA RTX 5090 Laptop |
| **VRAM total** | 24,463 MiB (≈23.9 GB) |
| **VRAM effective budget** | ~22 GB (keep 2 GB headroom for driver + display + CUDA context) |
| **NVIDIA driver** | 595.79 |
| **CPU** | Intel Core Ultra 9 285HX |
| **CPU cores / threads** | 24 cores / 24 threads |
| **System RAM** | 64 GB |
| **Disk free (C:)** | ~2.8 TB available (5.6 TB total) |
| **OS** | Windows 11 Pro 10.0.26200 |
| **Shell** | bash (git-bash) — Unix syntax, forward slashes |
| **Python** | `python` (not `python3`); dev in venv |

---

## 2. VRAM budget (the hard ceiling)

**Hard ceiling: 22 GB VRAM in use at any one moment.** Anything above that risks OOM and lost runs.

Practical allocation patterns:

| Pattern | VRAM split |
|---|---|
| Single-model probe | 1 model ≤ 20 GB, 2 GB activations / scratch |
| Two-model alignment (e.g., RSA, CKA cross-model) | 2 × ≤ 10 GB models, 2 GB scratch |
| Three-model universality sweep | 3 × ≤ 6 GB models, 4 GB scratch |
| Activation extraction (one model) | ≤ 18 GB model, 4 GB activations/hooks |

Quantization ladder (from the umbrella MODEL_DIRECTORY.md):

- **<1B params** — FP16 / BF16 (full precision)
- **1–7B** — Q6–Q8 (light)
- **7–30B** — Q4_K_M / Q5_K_M (medium)
- **30B+** — Q3_K / Q4_K_S (aggressive)

**Log quantization choice in every ledger entry.** A result at Q4 is not comparable to a result at FP16 without an explicit ablation.

### What fits

- Qwen3-0.6B at FP16: ~1.3 GB
- Pythia-410M at FP16: ~0.8 GB
- Mamba2-780M at FP16: ~1.6 GB
- Qwen3-4B at Q8: ~4 GB
- Qwen3-8B at Q4_K_M: ~6 GB
- DeepSeek-V2-Lite (16B, 2.4B active) at Q4_K_M: ~10 GB
- Qwen3-32B at Q4_K_S: ~22 GB (fills the envelope — no headroom for activations; avoid unless single-pass inference only)

### What does NOT fit on this machine

- Any model > 32B dense at Q4+
- Kimi-K2 / DeepSeek-V3 at precision > Q3 (require ≥ Q3_K_S and still flirt with OOM)
- Full-precision training of any model ≥ 3B params
- More than one 10+ GB model loaded concurrently

**If an experiment requires OOE (out-of-envelope) compute, the design gate stops. The options are: (a) shrink the model / use stronger quantization, (b) swap for an in-envelope substitute, (c) cut the experiment. Cloud is not available on this project.**

---

## 3. CPU and RAM budget

- **System RAM hard ceiling: 56 GB in-process** (keep 8 GB for OS + browser + editor).
- Parallel data-loading should respect `num_workers=0` on Windows + CUDA (see §5).
- For embarrassingly parallel CPU work (TwoNN on a large activation matrix, persistent-homology barcodes, etc.), use up to **20 threads**; leave 4 for OS and background tasks.

---

## 4. Disk budget

- **Free space budget: keep ≥ 500 GB free at all times** to avoid Windows tanking under low-disk pressure.
- Model weights go in the shared cache (`%USERPROFILE%\.cache\huggingface\hub` or the project's model-cache directory); do NOT re-download per experiment.
- Activation dumps for the atlas: partitioned per-system, per-layer, per-seed. A single 7B model dumping residual-stream activations at all layers for 10k tokens ≈ 5–10 GB; plan accordingly.
- `.gitignore` must exclude `*.npz`, `*.npy`, `*.pkl`, `*.pt`, `*.h5`, model weights, HF cache.

---

## 5. Windows + CUDA constraints (hard rules)

- **Python executable**: `python` (not `python3`)
- **PyTorch DataLoader**: `num_workers=0`, `pin_memory=False`. Multiprocessing with CUDA is unreliable on this machine.
- **sklearn / joblib**: `n_jobs=1` when any CUDA context is active. `n_jobs=-1` causes CUDA + multiprocessing deadlocks.
- **Encoding**: ASCII-only in source files (Windows cp1252 default breaks on Unicode).
- **Stdout buffering**: launch long runs with `PYTHONUNBUFFERED=1` so stdout reaches the ledger in real time.
- **CUDA state does not survive hibernate.** Any long experiment must save incrementally (per-system, per-seed) so a mid-run crash or sleep event doesn't lose the full sweep.
- **Allen Neuropixels access**: `remfile + h5py + dandi`. Do NOT use `allensdk` — not Python 3.13 compatible.

---

## 6. Time budget

- **An experiment should target ≤ 4 hours wall-clock** on this machine. Longer runs risk hibernate / OS update / power events that corrupt the run.
- Longer sweeps must be **checkpointable and resumable**. Design the save/resume path into the experiment before launch.
- Anything projected to run overnight: add explicit checkpoint-every-N-systems logic, verified on a smoke test.

---

## 7. What this constraint means for the atlas

Every primitive's agnosticism gate (3+ system classes) must be designed so that all three classes fit into the 22 GB envelope simultaneously OR can be run sequentially with disk-persisted activations feeding a unified analysis step.

Every Level-1 claim's causal test must use in-envelope models. No Level-1 claim can depend on a model we cannot actually load.

Every biological-validation step (Allen Neuropixels, fMRI) runs on CPU + RAM — no GPU needed for the current primitives. These steps are not compute-constrained; they are I/O and methodology-constrained.

**Design implication:** the atlas is built from a vocabulary of small-to-mid models (≤ 8B dense at Q4; ≤ 32B MoE with ≤ 8B active at Q4). Frontier-class models appear only as sanity checks, not as primary atlas entries. This aligns with the manifesto — efficiency through geometry, not scale.

---

## 8. When the envelope must grow

If a sacred outcome genuinely cannot be achieved in-envelope, the escalation path is:

1. **Document the requirement explicitly** — which experiment, which primitive, which system class, exactly how much VRAM / RAM / time, and why a smaller substitute cannot serve.
2. **Codex design-gate review** of the requirement. If Codex agrees the substitute cannot serve, the constraint is recorded as a *known scar* on the atlas.
3. **User decision** — only Devansh authorizes out-of-envelope work. Cloud compute is not currently available; any authorization here means either a hardware upgrade or an explicit scope reduction of the scientific claim.

Never quietly exceed the envelope. Never assume a "small overage" is fine — Windows + CUDA under memory pressure loses runs silently.

---

## 9. Compliance checklist (for Codex prompts and experiment preregs)

Every Codex design-gate prompt and every preregistration must answer:

- [ ] Max concurrent VRAM usage during this experiment? (must be ≤ 22 GB)
- [ ] Max system RAM usage? (must be ≤ 56 GB)
- [ ] Wall-clock estimate? (should be ≤ 4 h; if longer, checkpoint plan required)
- [ ] Disk footprint for artifacts? (activation dumps, model cache)
- [ ] Quantization level per model used? (logged in ledger)
- [ ] Save-resume path verified on a smoke test?

If any box is unchecked at the design gate, the experiment does not proceed.

---

*This file is the binding envelope. When hardware changes (new GPU, more RAM, etc.), update this file first — in its own commit — before any experiment plan references the new capacity.*
