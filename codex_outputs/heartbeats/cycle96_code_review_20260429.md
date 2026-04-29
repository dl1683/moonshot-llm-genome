**Findings**

- **S10** [code/genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:1206>): `compute_verdict()` gates every `loao_results` entry. Since main now passes A/B/C/D/E into it, telemetry and Shesha controls are accidentally required to satisfy PASS criteria. This violates the prereg A/B co-primary logic and can turn a valid geometry result into FAIL/WEAK PASS for the wrong reason.

- **S9** PID 2008 is unaffected, but that also means it will not use these fixes. Process start was `2026-04-29 12:51:45Z`; `genome_182_triage_arena.py` and `genome_180_forecast.py` were modified later around `13:29Z`. Python loaded the old code at process start. Letting PID 2008 finish is safe for training, but do not trust its final analysis verdict without rerunning analysis from saved cells.

- **S9** [code/genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:1290>): current Shesha augmentation path is not valid: it references missing `ARCH_CONFIGS[*]["hf_id"]`, skips `seq_kd_full`, and replays `embed_anchor` as plain CE without the anchor loss. That does not implement the cycle95 Shesha residual kill.

- **S8** [code/genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:522>): Model C has no loss/gradient/trajectory telemetry, but it is not strictly “only manifold/spectral geometry.” It includes `hidden_norm_early_late_ratio` and `hidden_var_early_late_ratio`, which are activation scale/variance ratios, not spectral/manifold invariants.

- **S8** [code/genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:1055>): AUROC delta is compared only against the baseline selected by best MSE, not the best AUROC baseline. The prereg says AUROC must beat the best baseline; this can overstate bad-run detection.

- **S7** [code/genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:1177>): `simulated_kill()` divides survived label sum by total label sum. With negative gains, this metric can become nonsensical while still entering PASS checks.

**Verified**

Scratch exclusion is correct in the current source: scratch rows are used only to build denominators, skipped at label construction, and all LOAO calls use `labeled`. I found no later analysis path that re-adds scratch rows.

C/D feature lists are disjoint and exactly cover Model B: `B=16`, `C=10`, `D=6`, overlap `0`, missing `0`.

Performance: adding C/D roughly doubles analysis work but not materially versus training. Synthetic 48-row benchmark with actual constants: A `22.7s`, B `22.2s`, C `22.0s`, D `21.7s`, total `88.6s`. So C/D add about `44s`; current E would add another ~`22s`. Training remains the bottleneck.

