**Q1**
- Keep the canonical recipient as random-init `Qwen3-0.6B` architecture. `minimal_3L_noMLP` from `g141` is a Llama-family student with different shapes/tokenizer, so the full-weight Frobenius anchor in `g165` stops measuring the same object. That turns `g165` into cross-arch distillation, not a clean washout test.
- Do not switch the canonical donor to `Qwen3-1.7B`. Full Frobenius anchoring assumes parameterwise alignment; `1.7B` breaks that. A stronger-teacher test belongs on the `g154/g155` distillation track, not `g165`.
- Yes, add a hard-cut arm. In the current code’s step numbering, that means anchor active at training `step=1`, then zero from `step=2` onward. That is the cleanest test of “early-help only, no continued anchor.”

**Q2**
- The better framing now is: zero-step donor signal is already real, but persistence under continued adaptation currently appears only when donor-carrying weights are prevented from drifting. `g125` is the decay-rate-zero boundary condition.
- That argues to keep `g165`, not replace it. `g165` is the interpolation test between `g125`-style zero-decay persistence and the 5/7 washout cases.
- A deeper `g125` sweep is lower upside. It will likely refine a tiny `+0.07` endpoint effect; `g165` still has the only plausible path to adaptive persistence, which is the more competitive result.
- The real blind spot is not recipient size. It is anchor submanifold: `g125` persisted with donor attention frozen, while `g165` currently anchors all weights.

**Q3**
- For the locked `g155` flagship claim, the integrity bar is hard-blocking. The prereg explicitly disallows the headline if wall-power is missing.
- `nvidia-smi` GPU power plus CPU package power is still worth collecting, but only as a caveated proxy / decomposition / dry run. It cannot stand in for official `TEI/kJ`.

**Score**
- Expected queue §0.1 uplift after the cycle-27 correction: `6.6/10`.

**One change**
- Add one extra `g165` arm before lock: `attention-only` anchor, `lambda_0 = 1.3e-3`, active only at training `step=1`, zero from `step=2` onward. That directly tests the corrected `g125` boundary condition inside `g165` without replacing the canonical experiment.