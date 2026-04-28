Updated [genome_173_cross_arch_flop_cashout.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_173_cross_arch_flop_cashout.py:1>) in place.

It now runs 6 arms x 3 seeds with scheduled late-KD arms, resume support, schedule-aware KD compute, revised pass/fail criteria, and per-architecture `late_kd_vs_full_kd` JSON summaries.

Verification passed: `python -m py_compile code/genome_173_cross_arch_flop_cashout.py` and `git diff --check`; I did not launch the 18-cell training run.