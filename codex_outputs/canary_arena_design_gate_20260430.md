Created the prereg draft here:

[research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md:1>)

Key locked choices:
- 10 lm_head conditions x 3 seeds = 30 cells, staying inside the 4h envelope.
- Untied model, lm_head init only, no active anchors.
- Step 0 and step 50 lm_head geometry features only for the primary Ridge model.
- Primary comparator is scalar step-50 validation NLL.
- PASS requires >=25% LOCO MSE reduction vs early loss, CI lower bound > 0, held-out R2 >= 0.35, permutation p <= 0.05, and geometry beating loss in at least 8/10 held-out conditions.
- Section 0.1 score: PASS -> about 6.9/10; WEAK_PASS -> about 6.2/10; FAIL -> cap/drop near 5.4/10.

Validation: the draft is 997 words and ASCII-only. I did not modify `WIKI.md`; it is currently dirty from an unrelated one-line change in the worktree.

