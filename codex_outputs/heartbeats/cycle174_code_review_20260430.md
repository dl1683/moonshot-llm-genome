**Findings**
- SEV-5: `compute_verdict` over-labels `FAIL`. The prereg says `FAIL` only when no arm reaches `+0.10`, but the code defaults to `FAIL` for any non-pass pattern and never checks max arm gain. This can misclassify weak-positive or mixed results. See [compute_verdict](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:262>) and prereg [criteria](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_195_untied_input_output_factorial_2026-04-30.md:53>).

- SEV-4: The g194 gate is not enforced in code. `g195` will start regardless of whether `results/genome_194_scalar_direction_factorial.json` is complete or says `PASS_DIRECTION`. Current live g194 result file still says `INCOMPLETE`, though the signal may be locked analytically. See [g195 main](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:293>) and [g194 verdict](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_194_scalar_direction_factorial.json:293>).

- SEV-2: Smoke verdict will always be `incomplete` because `compute_verdict` hardcodes global `SEEDS=[42,7,13]` instead of the payload/run seeds. Full 15-cell run is unaffected. See [seed check](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:247>).

**Checks**
- Full run should execute: `py_compile` passed, and a synthetic 1-step pass through all 5 arms completed without shape/device/anchor errors.
- No data leakage found between arms. Models and optimizers are rebuilt per cell; seeds are reset; shared numpy anchors are read-only in practice.
- Resume is correct at cell granularity: completed cells skip, crashed in-progress cell reruns from scratch. No mid-cell checkpointing.
- VRAM is comfortable on this machine: tied `141.7M`, untied `193.1M`; local GPU is `23.9 GiB`. Prereg’s `~1.6 GB` estimate is low for AdamW training state, but not launch-blocking.