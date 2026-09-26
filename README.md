# Neural Dissection Lab

> Da Vinci opened corpses to understand anatomy. We open neural networks to
> understand how they learn, what they store, and how they break. No thesis to
> defend, no product to ship — dissection, curiosity first.

## Mission

Train small neural networks (1M–100M parameters; sweet spot 1–10M), then:

- **watch them learn** (loss landscapes, phase transitions, grokking, curriculum),
- **see what they store** (which layers/heads/neurons hold what),
- **make them forget** (selectively erase a skill; measure collateral damage),
- **transplant their organs** (swap layers/heads/MLPs between nets),
- **evolve them** (lineage and mutation instead of, or with, gradient descent),
- **poke preloaded nets** and change behavior in *predicted* directions.
- **SEE the system** (standing VISUALIZER thread): representations and
  visualizations are discovery instruments — the right picture reveals
  structure that metrics hide, and what can be represented can be
  manipulated. Token journeys, attention maps, lesion atlases, write-space
  geometry, prediction evolution through depth. Every review cycle includes
  visualization thinking; `lab/vNNN_*.py` + `runs/vNNN/` are the viz series.

The method is play: run hundreds of experiments, graph everything, follow
anomalies, let the good questions emerge from the anatomy.

## Why small networks

A 1–10M-param transformer trains in minutes on one RTX 5090 Laptop. Iteration
is effectively free, experiments are disposable, and a dissection program lives
or dies on iteration count. Anything over 100M params is out of scope.

## What we have learned (day one + two; see DAY_ONE_REPORT.md and THINKING.md)

The lab's first 48 hours produced seven graded laws about a 2.7M char
transformer — each with its evidence chain, replication stamps, and known
confounds (card v3, THINKING.md T010-T023):

1. **A causal gate exists in every net tested ≥6 layers** (failed the
   strict criterion at 4L, where the mode is the final block); its depth
   varies with seed, regime, and architecture budget (T012/T014/T020).
2. **Anatomy is plastic; damage tracks energy** — lesion maps reorganize
   under constraint at parity loss; the late-MLP energy carrier replicates
   5/5 nets. [0.84M, n=1, T023: an equalized-write net passes parity
   (1.515 vs 1.537) with attention/calibrator unchanged — schedule
   possibly decorative, unreplicated.]
3. **The residual-stream basis is init-anchored** — the alignment ladder
   1.0 → 0.53 → 0.15 → 0.00; stream-facing matrices are the violent
   grafts (T003/T006/e029/e031/e041).
4. **Far-retrieval is task-elicited, not architectural** — none on natural
   data; a refrain-density threshold (5-20% by the strict shared-probe
   read; R8-flagged: the p0 interference magnitude is partly net-quality
   artifact) flips interference into retrieval; retrieval heads form
   discretely in the late-attention slot (2.7M series; 10M leg
   suggestive).
5. **First-order ascent cannot selectively forget** (r=1.09
   memorization-symmetric at the bar; every granularity tested).
6. **Entity knowledge is address-plus-body** — 384-parameter row surgery
   damages a name at S_name 573 for +0.0008 nats; complete erasure was
   net-specific luck (T016).
7. **The edit law: address, ability, expression, history** — removal is
   surgical, installation is plastic-but-never-surgical, expression needs
   free-shaped exposure (teacher-forced install is constitutionally
   silent), and re-learned memories differ in route and surgical
   resistance (T015/T018/T019).

**The meta-law:** the laws are ensemble properties; the mechanisms are
samples. Small nets are degenerate ensembles — WHICH component carries a
function is a seed lottery; THAT the coarse allocation exists is forced.
Method: mechanism claims enter the card at H only after ≥3 nets.

In progress: e040 (is init-anchoring evolvable under lineage selection?).

## History (tombstone commit `106aeff`, 2026-09-24)

Three prior programs live in git history below that commit:

1. **Neural Genome** (Apr–May 2026): transplanting trained capability between
   models. Strong-form transfer falsified; one real finding survived — *output
   lm_head token-row directions are causal training coordinates*.
2. **LLM control surfaces** (May–Jul 2026): MC001–MC033, KSQ001–015. 0 promoted
   mechanism cards; 1 bounded card (attention-write mediation in Qwen3-1.7B).
3. **HANDLE** (Aug 2026): private-state substrate proposal; NO-GO; task built,
   never trained.

Doctrine inherited from those eras: **a past result is a warning about
experimental design, not evidence.** Two honesty reflexes survived (see Rules
7). Everything else — preregistration courthouse, mechanism cards, adversarial
review bureaucracy — was deliberately left behind. We are playing, not litigating.

## Rules of the lab

0. **80% thinking, 20% doing.** Results are cheap; wrong conclusions are
   expensive. Every result gets a THINKING.md entry — multiple alternative
   explanations, discriminating observations, registered predictions — BEFORE
   the next experiment builds on it. **No interpretation, no next experiment.**
   An idle GPU during thinking is the lab working correctly. When in doubt,
   think more and run less; the queue can wait, understanding compounds.
1. **Play first.** Questions sharpen after looking. Boring hypotheses about
   what to find are not required in advance.
2. **One file per experiment:** `lab/eNNN_name.py`, self-contained, seeded,
   runnable as `python lab/eNNN_name.py` from repo root.
3. **Every run writes `runs/eNNN/`** with `metrics.json` and at least one PNG
   graph. No graph, no experiment.
4. **Every experiment gets a NOTES.md entry:** what we did / what we saw /
   what's next. If it isn't written down, it didn't happen.
5. **Commit constantly.** Small commits beat big ones. Git is the memory.
6. **Never delete `runs/` or `data/`.** Disk is cheap; forgetting is expensive.
7. **Honesty reflexes (not bureaucracy).** Before believing any internal-state
   finding, ask: (a) do logits/behavior alone already predict it? (b) does
   intervening on it actually change behavior? If not, it's a curiosity, not a
   discovery.
8. **Time budget:** any single experiment step ≤ ~30 min. If slower, shrink the
   model or the data.
9. **Compute envelope (2026-09-25 shutdown incident — permanent):** models
   ≤1M params by default (5M absolute ceiling, only with explicit cause).
   Keep ≥15% GPU headroom (util ≤85%, memory ≤85%). Check
   `lab/common.py: gpu_status()/gpu_ok()` BEFORE any launch; insert
   `cooldown(60–120s)` between training runs; no concurrent GPU jobs.
   Prefer batch 32 and ≤180s training caps. Thermal guard: no new launches
   above 80°C; wait for ≤65°C when heat-soaked.
9. **Parallel by default.** Keep the GPU busy with the main experiment AND run
   1–3 background subagents (exploration, analysis, literature of the lab)
   simultaneously. Multiple things move at once or the lab is underutilized.
10. **Checkpoint everything.** `train_model` writes a resumable snapshot at
    every eval point. Interrupted runs resume from checkpoint — never retrain
    from scratch. A shutdown should cost minutes, not hours.

## Standing question source: neuro-ai-lab

`C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab` (see
`reference/NEURO_AI_LAB.md` for the distillation) is a sibling project whose
brain/biology-inspired mechanisms are a permanent source of dissection
questions: does a tiny ANN have an analogue of mechanism X? Can we install
one? Would it make the net better or worse — and how would we *prove* the
analogue is real rather than a metaphor? Every hourly review should re-mine
it.

## Cadence — the heartbeat system (this is how the lab never idles)

One cron automation (every 10 min, `*/10 * * * *`) drives everything.
**The heartbeat's core purpose is a FLEET-LIVENESS CHECK**: every
heartbeat verifies that background agents are actively working — either
experiment executors or thinking/review/visualization subagents
(interpreter / ideator / researcher / critic / visualizer angles). A
heartbeat that finds zero agents running and dispatches none is a FAILED
heartbeat, even if the ledgers are clean. The live roster is recorded in
`STATE.json.current_experiment` every heartbeat. GPU idling is fine —
thinking agents are work too.

| Condition | What it does |
|---|---|
| every 10 min | **Heartbeat = fleet check**: enumerate running agents; if the fleet is empty, dispatch immediately (top READY experiment, or interpreter/ideator/researcher/visualizer/critic on the freshest material); harvest anything that landed; log results in NOTES.md; commit |
| `last_review` > 60 min old | **Frontier review**: 3 parallel subagents (auditor / ideator / critic) review the notebook and queue, inject new ideas, retire stale lines; timestamped entry in REVIEWS.md; resets `last_review` |
| `last_novelty` > 2 h old | **Novelty guarantee**: a genuinely new experiment line starts (not a continuation); resets `last_novelty` |

All timestamps live in `STATE.json` (`last_heartbeat`, `last_review`, `last_novelty`).
Any agent — scheduled or fresh — reads that file first and treats stale values as
action items ("review overdue → run it now"). Only one automation exists per
session; do not create more, the trigger logic above replaces a second clock.

The cadence is subordinate to Rule 0: heartbeats may spend their turn
thinking (writing THINKING.md) instead of launching runs, and the GPU staying
idle while an interpretation is pending is correct behavior, not a failure.
The measure of a good session is insight per experiment, not experiments per
hour.

## Repo map

```
lab/        experiment code (one file per experiment; common.py = shared harness)
runs/       outputs: runs/eNNN/metrics.json + PNGs + samples (never delete)
data/       corpora
NOTES.md    lab notebook (append-only, newest at top)
QUEUE.md    experiment queue + idea parking lot (reviews feed this)
REVIEWS.md  hourly frontier-review log (timestamped)
STATE.json  machine-readable current state + cadence timestamps
AGENTS.md   one-page resume protocol for any fresh agent/thread
```

## Resume protocol (for a fresh agent, new thread, or post-compaction)

Read, in order: `README.md` → `STATE.json` → top of `NOTES.md` → `QUEUE.md`.
Then either finish the experiment named in `STATE.json.current_experiment` or
start the top of the queue. Follow the cadence rules. Commit. That's the whole
job.
