# Heartbeat protocol (Devansh, 2026-04-26)

This file specifies the `/loop` heartbeat for the Neural Genome moonshot. It is loaded by the loop prompt at every fire.

## Cadence

Heartbeat fires every ~25 minutes (self-paced). Each fire is one of two types:

- **Activity heartbeat** (heartbeats #1, #2, #4, #5, #7, #8, ... — 2 of every 3)
- **Codex review heartbeat** (heartbeats #3, #6, #9, ... — 1 of every 3)

The script tracks the cycle counter in `experiments/heartbeat_cycle.txt` (one integer, incremented each fire).

## Activity heartbeat checklist (every fire)

The agent must, in order:

1. **Verify running experiments are alive.** `tasklist | grep python` to check Python processes; tail `results/genome_*_run.log` for any experiment whose status in WIKI says RUNNING. If a process died unexpectedly, immediately diagnose (likely Unicode bugs, OOM, or NaN). Do NOT silently mark experiments as failed — investigate.

2. **Check for completed experiments.** Any `results/genome_*_run.log` whose tail shows "Saved:" or a verdict line means the experiment finished. Read its JSON, write a ledger entry, update WIKI + EXPERIMENTS.md + CLAIM_EVIDENCE_MAP, commit + push.

3. **Launch the next queued experiment if GPU is free.** Free = no python.exe in tasklist consuming GPU memory. Pre-staged queue is in WIKI (the highest-leverage one runs next).

4. **Audit task list.** Any TaskCreate from prior fire that is still pending? Move to in_progress or delete if stale.

5. **Mission alignment.** Are active tasks serving the §0 end goal (capability transfer + electricity-grade efficiency demo)? If drift detected, redirect or fire Codex.

## Codex review heartbeat (every 3rd fire — additionally)

Beyond the activity checklist, the Codex-review heartbeat fires TWO Codex consults in parallel:

### A. Codex Code Reviewer (Correctness Engineer + Performance Engineer)

```
codex exec --full-auto --skip-git-repo-check -C <PROJECT_DIR> -o codex_outputs/heartbeat_code_review_<TIMESTAMP>.md "<PROMPT>"
```

Prompt: "You are Correctness Engineer + Performance Engineer for the Neural Genome moonshot. Read CLAUDE.md, WIKI.md (especially the most recent activity), and any code/genome_*.py modified in the last 4 hours (use git log --since='4 hours ago' --name-only to find them). Audit for: bugs, edge cases, NaN risks, OOM risks, encoding issues (Unicode-on-Windows like the g148/g156 Δ crash), silent-data-drop bugs (like g156 Severity-8). For any severity ≥7 finding, give the exact code change. Report under 500 words. If clean, say so."

### B. Codex Direction Reviewer (Architecture Theorist + Competitive Analyst)

```
codex exec --full-auto --skip-git-repo-check -C <PROJECT_DIR> -o codex_outputs/heartbeat_direction_review_<TIMESTAMP>.md "<PROMPT>"
```

Prompt: "You are Architecture Theorist + Competitive Analyst for the Neural Genome moonshot. Read CLAUDE.md §0/§0.05/§0.1, WIKI.md, research/CLAIM_EVIDENCE_MAP.md, research/programs/post_g156_pass_program.md, and the latest experiment results in results/. Are we on the right breakthrough-axis trajectory or drifting into patch-old-chain work? Should the next queued experiment be reordered? Is there a higher-leverage move available given the most recent state? Score the active queue's §0.1 expected uplift (1-10) and recommend ONE concrete change if expected uplift < 6. Report under 400 words."

Both consults run in parallel via `run_in_background: true`. Their outputs are read on the FOLLOWING heartbeat (since Codex takes 5-15 min) and integrated.

## Output format per heartbeat

After the checklist + (optionally) firing reviews, post to chat:
- One line: cycle counter, type (activity / review), GPU status, running experiments
- Bullet list: any actions taken (commits, launches, integrations)
- One line: next queued action

Keep under 200 words per heartbeat output.

## What "actively working" means

Between heartbeats, the agent must NOT idle. If there is no GPU work and no Codex consult to integrate, the agent should:
- Pre-stage the next experiment's prereg
- Audit a canonical doc for staleness
- Do an anti-entropy pass
- Cross-check the ledger against results/

NEVER end a turn with "scheduled wakeup" or "waiting for X to finish" unless ALL parallel work axes are blocked. The wakeup is a last resort, not the pace.
