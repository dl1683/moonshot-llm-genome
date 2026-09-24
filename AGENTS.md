# AGENTS.md — read this first, then act

You are entering the **Neural Dissection Lab**. Someone (human or scheduled
automation) sent you here to keep the work going. You do not need any
conversation history to operate.

## Read in this order (5 minutes)

1. `README.md` — mission, rules, cadence. The constitution.
2. `STATE.json` — what is running now, and the cadence timestamps
   (`last_heartbeat`, `last_review`, `last_novelty`). If any timestamp is
   stale (>15 min heartbeat / >75 min review / >2 h novelty), treat that as
   your first action item.
3. `NOTES.md` (top entries) — what was just done and what it showed.
4. `QUEUE.md` — what to do next.
5. `REVIEWS.md` (last entry) — the most recent frontier-review decisions.

## Then do exactly one of these

- **An experiment is running** (fresh lock/mtime in `runs/`) → let it finish,
  then record results in NOTES.md and commit.
- **Nothing is running** → start the top READY experiment from QUEUE.md, or the
  one named in `STATE.json.current_experiment`.
- **Review is overdue** → run the frontier review (see README cadence table):
  3 parallel subagents (auditor / ideator / critic), write REVIEWS.md entry,
  update QUEUE.md + STATE.json, commit.

## Hard rules

- One file per experiment in `lab/`, outputs in `runs/eNNN/` with metrics.json
  + PNG graph. NOTES.md entry for every experiment.
- Models ≤100M params; prefer 1–10M; single steps ≤30 min.
- Never delete `runs/` or `data/`.
- Commit constantly (git is the lab's memory).
- Honesty reflex before believing a finding: do logits/behavior alone predict
  it? does intervening change behavior?
- Do not create new automations; the two crons (heartbeat, hourly review)
  already exist. Just do the work.
