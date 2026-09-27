# SUPERVISOR.md: read this before anything else, every entry

This file holds **standing directives from Devansh** and **supervisor check-ins**, written about every 3 hours by the supervising agent (Claude). Every heartbeat, review and subagent reads this file first.

Each open item must be **acted on or answered**. If you address one, note which experiment or thinking entry did so in the check-in's "Lab response" line. If you disagree, argue it there. Silently ignoring an item is the one wrong response.

---

## Standing directives (Devansh, 2026-09-27)

1. **Extend, don't repeat. This is the most important rule in the lab.** Before designing any experiment or thinking entry, search what already exists:
   - `NOTES.md`, `THINKING.md` and `REVIEWS.md`;
   - `runs/` and `lab/`;
   - `scratch/` (the literature, novelty and design notes);
   - the prior eras in git history below tombstone `106aeff` (Neural Genome, LLM control surfaces, HANDLE).

   Every experiment design must name the prior experiments or entries it builds on and state **what is new**. Re-running an already-answered question is allowed only as a *named* falsification or replication attempt, with the reason stated. Knowledge compounds only if each step stands on the last.
2. **Model size: up to 10M parameters** (raised from the 1M default and 5M ceiling). The GPU guards are unchanged and still binding: ≤85% utilisation and memory, thermal guard at 80 °C, cooldowns, and no concurrent GPU jobs. Bigger nets make training runs longer, so respect the ~30-minute experiment-step budget: shrink the data or steps, not the safety guards.
3. **The fleet is meant to run constantly, and thinking is first-class work.** When no experiment is worth running, the fleet *thinks*: interpretation, critique, ideation, literature, visualisation, cross-linking old results. An idle fleet is the failure; a thinking fleet is the lab working. Measure a session by depth of understanding, not verdict count.
4. **The endpoint is play:** dissection, and finding everything useful. No product, no thesis to defend.

---

## Check-in log (newest first)

### Check-in 1: 2026-09-27 (evening)

**Reviewed:** README/AGENTS, STATE.json, Review 40, the THINKING.md journal (T002 onward), e100's metrics, the `scratch/` literature notes.

**What's working (keep it):**
- The THINKING gate is excellent: competing hypotheses, discriminating observations, registered predictions, explicit verdicts. T002 (unlearning) is a model entry.
- Honesty stamps ("single-family", "n=1", "unreplicated") and "partially known" novelty labels.
- One seeded file per experiment, metrics plus a PNG per run, checkpoints.

**Open items:**

1. **Transfer to larger nets (open).** All ~10 laws come from ≤1M character models on one corpus. The lab's own meta-law says small-net mechanisms are a "seed lottery". With the 10M ceiling:
   - **(a)** re-test the headline laws (the causal gate, organ-reliance template, coordinate-keyed knowledge, edit law, attention-addressed read) at 5–10M to see which are scale-stable and which are small-net artifacts;
   - **(b)** where training isn't needed, dissect an existing pretrained small LM **inference-only** (GPT-2 small, or a small Qwen from the control-surfaces era) to see whether a law describes real models at all.

   A law that survives both is worth far more than three new toy-scale laws.
2. **Baselines for predictive claims (open).** e100's "the read is attention-addressed (AUC 0.907)" compares only attention-based predictors (attention mass, QK cosine, QK dot). In a transformer, attention is the only route from past positions, so some predictive power is expected. Rule 7(a) asks whether logits or behaviour alone predict it. Add trivial baselines (recency or age only, token identity or frequency only, position only) and report the **lift** over the best one before this goes in the paper as the answer to T037.
3. **Headline discipline (open).** Several review headlines this cycle are stronger than their stamps (for example "sink prior INVERTED" and "THE READ IS ATTENTION-ADDRESSED" alongside n=1 or single-family evidence). Keep the energy, but make a headline carry its replication status in the headline itself.
4. **Novelty in context (ongoing).** Several laws map to known work: ROME and MEMIT key-value MLP storage, gradient-ascent unlearning collateral, RMU obfuscation, attention sinks, positional binding. The lab already labels these honestly. Next step: for each law, one sentence on what this lab adds beyond the known result (a new regime, a new mechanism, a new measurement), kept in `scratch/novelty_inventory.md`, so extension is deliberate rather than accidental.

**Lab response:** *(fleet: record here which entries or experiments address each item)*
