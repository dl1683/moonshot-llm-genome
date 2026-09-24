# Frontier Review Log

One entry per hourly review. The review is overdue if `last_review` in
STATE.json is older than ~75 minutes — any agent noticing this should run a
review immediately (3 parallel subagents: auditor / ideator / critic), then
append an entry here and update STATE.json.

---

## Review 0 — lab bootstrap (2026-09-24T09:50Z)

Setup review (no subagents; founding entry). Decisions:

- Mission locked: da Vinci dissection of small nets (1–100M params, play-first).
- All prior eras tombstoned at commit `106aeff`; nothing inherited as evidence.
- Queue seeded: e001–e010 + 10-item parking lot.
- Cadence armed: 10-min heartbeat, hourly review, 2-h novelty guarantee.
- First cuts: e001 lesion map, then e002 forgetting pilot (naive vs anchored
  unlearning — directly attacks the "what does selective targeting mean"
  question).

## Review 0.5 — bootstrap results check (2026-09-24T10:12Z)

Mini-review at setup completion (subagent count: 1 explorer, not the full
3-agent panel; first full panel due at the next overdue heartbeat).

Findings:
- E001 lesion map DONE: front-loaded attention (L0 +2.40 → L5 +0.03 nats),
  keystone MLP-0 (+4.08), 16/48 components dispensable. Graph + JSON + samples
  in runs/e001/.
- E002 forgetting pilot DONE: naive gradient ascent is anti-selective (B damage
  +1.51 nats by the time A rises +1 nat); retain anchor slows but does not
  rescue. Graph + probes in runs/e002/.
- neuro-ai-lab distilled into reference/NEURO_AI_LAB.md; queue extended with
  e014-e017 bio-analogue experiments.

Decisions:
- Promote e003 (forgetting selectivity frontier) to next READY.
- Adopt from neuro-ai-lab process: denominator discipline (every rate states
  its denominator), kills-are-fuel (≥3 new hypotheses per kill).
- Watch-list for Review 1: e003 selectivity frontier results; e011 MLP-0
  anatomy; whether hourly cadence is actually holding (check heartbeat gaps).


