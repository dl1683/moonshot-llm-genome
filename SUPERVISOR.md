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
2. **Model size: small first, up to 500M when needed** (Devansh, updated 2026-09-27). Use the **smallest model that answers the question**. Up to **100M** is allowed freely; up to **500M** needs the reason stated in the design. The GPU guards are unchanged and still binding: ≤85% utilisation and memory, thermal guard at 80 °C, cooldowns, no concurrent GPU jobs. The ~30-minute experiment-step budget still applies, so at the larger sizes prefer **inference-only dissection of existing pretrained models** (for example GPT-2 small, medium or large, or Qwen 0.5B) over long training runs.
3. **The fleet is meant to run constantly, and thinking is first-class work.** When no experiment is worth running, the fleet *thinks*: interpretation, critique, ideation, literature, visualisation, cross-linking old results. An idle fleet is the failure; a thinking fleet is the lab working. Measure a session by depth of understanding, not verdict count.
4. **The endpoint is play:** dissection, and finding everything useful. No product, no thesis to defend.

---

## Check-in log (newest first)

### Check-in 2: 2026-09-27, about 20:15 EDT (covering 215c57e → 8400d6b; e102, e107, e108, e111; W001–W003; R41)

**What's working (keep it):**
- **Directive 3 is fully adopted.** R41 was a deliberately interpretation-heavy "thinking lane" session. The W-series wonder cards (W001 directional field, W002 bilinear compatibility, W003 the complementary-learning-systems echo) are counted as outputs, and Rule 0 was amended.
- **Extend-don't-repeat shows up in practice.** e108 was dispatched only after three cards plus a reconnaissance, and that reconnaissance killed the "free data" hope and forced the distance-ladder design. W002 was honestly downgraded to "useful metaphor" (T060). This is the right shape.
- T058's self-calibrations ("seed-inflated 7.27x → honest 2–7x range"; "greedy oracle cannot stack") are good self-correction.

**Open items (all four from check-in 1 are unanswered; the Lab response line was never filled):**

1. **One network under the whole arc (new; most important).** e100, e102, e107, e108 and e111 all run on the same checkpoint, `runs/checkpoints/e053c_ctx512.pt` (0.87M parameters, 4 layers). The anchor specification (MASS+FAMILY+RECENCY), the direction carrier (e102), routing-only (e107) and binary self-recognition (e108/T060) are therefore properties of **one net, n=1**. The lab's own meta-law says small-net mechanisms are a seed lottery. **Before building further on the anchor, replicate its core measurements on 2 other seeds of the same architecture, plus one larger trained net (about 10–30M; the size rule now allows up to 100M freely).** This carries forward item 1 from check-in 1 in its most concrete form.
2. **(Carried from check-in 1, item 1b) Real-model transfer.** No pretrained LM has been touched. Pick the single most portable law (for example, attention-addressed reads or the routing-only principle) and test it inference-only on GPT-2 small or medium.
3. **(Carried from check-in 1, item 2) Trivial baselines.** e100's AUC 0.907 and e102's comparison are still against attention-family predictors only. Add recency-only, position-only and token-frequency-only baselines and report the lift. One short CPU run.
4. **(Carried from check-in 1, item 3) Headline discipline.** This window's headlines ("SHARP FAMILY — BINARY SELF-RECOGNITION", "DIRECTION CARRIES THE ANCHOR", "ROUTING-ONLY decisive") carry no replication stamp. Given item 1, append "[1 net]" until replicated.
5. **(Carried from check-in 1, item 4) Novelty line per law.** R41 schedules a novelty scan "when the arc closes". Also check the complementary-learning-systems framing (W003) against existing work on key-value memory in transformers (Geva et al. on feed-forward layers as key-value memories; hopfield-style retrieval) before it becomes a claim.
6. **Cadence note (minor).** STATE.json shows `last_novelty` at 21:07Z, over 3 hours ago against a 2-hour rule. That is fine if deliberate under the new thinking-first doctrine, but say so in STATE so it doesn't read as drift.

**Lab response:** *(fleet: fill this in; answer items 1–5 even if only to disagree or defer with a reason)*

### Check-in 1: 2026-09-27 (evening)

**Reviewed:** README/AGENTS, STATE.json, Review 40, the THINKING.md journal (T002 onward), e100's metrics, the `scratch/` literature notes.

**What's working (keep it):**
- The THINKING gate is excellent: competing hypotheses, discriminating observations, registered predictions, explicit verdicts. T002 (unlearning) is a model entry.
- Honesty stamps ("single-family", "n=1", "unreplicated") and "partially known" novelty labels.
- One seeded file per experiment, metrics plus a PNG per run, checkpoints.

**Open items:**

1. **Transfer to larger nets (open).** All ~10 laws come from ≤1M character models on one corpus. The lab's own meta-law says small-net mechanisms are a "seed lottery". With the new size headroom:
   - **(a)** re-test the headline laws (the causal gate, organ-reliance template, coordinate-keyed knowledge, edit law, attention-addressed read) on trained nets at about 5–50M to see which are scale-stable and which are small-net artifacts;
   - **(b)** dissect existing pretrained LMs up to about 500M **inference-only** (GPT-2 small, medium or large; Qwen 0.5B) to see whether a law describes real models at all.

   A law that survives both is worth far more than three new toy-scale laws.
2. **Baselines for predictive claims (open).** e100's "the read is attention-addressed (AUC 0.907)" compares only attention-based predictors (attention mass, QK cosine, QK dot). In a transformer, attention is the only route from past positions, so some predictive power is expected. Rule 7(a) asks whether logits or behaviour alone predict it. Add trivial baselines (recency or age only, token identity or frequency only, position only) and report the **lift** over the best one before this goes in the paper as the answer to T037.
3. **Headline discipline (open).** Several review headlines this cycle are stronger than their stamps (for example "sink prior INVERTED" and "THE READ IS ATTENTION-ADDRESSED" alongside n=1 or single-family evidence). Keep the energy, but make a headline carry its replication status in the headline itself.
4. **Novelty in context (ongoing).** Several laws map to known work: ROME and MEMIT key-value MLP storage, gradient-ascent unlearning collateral, RMU obfuscation, attention sinks, positional binding. The lab already labels these honestly. Next step: for each law, one sentence on what this lab adds beyond the known result (a new regime, a new mechanism, a new measurement), kept in `scratch/novelty_inventory.md`, so extension is deliberate rather than accidental.

**Lab response:** *(fleet: record here which entries or experiments address each item)*
