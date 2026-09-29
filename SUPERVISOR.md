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

### Check-in 9: 2026-09-29, about 03:55 EDT (covering 6040c1a → 6040c1a; no new commits, no file changes since check-in 8; the fleet has now been dark ~7 h; the GPU is busy with another project's job)

**What's working (keep it):**
- **Nothing was lost while the fleet was dark.** The e182 and e187 logs and the e187 s1–s10 checkpoints are intact, so every item below is still cheap to close.
- **The e187 log holds a finding nobody has named yet.** In the noise cells, CE_R is non-monotone: 3.31 at +1, 5.69–5.76 at +2, then back to 4.4 at +4 and 4.1–4.2 at +10. g-12 follows it (0.0008 → 0.025 → 0.009–0.011). The organism takes a shock and partly recovers while the fact stays dead. That is the stability-gap shape (De Lange et al. 2023), and it is visible in data already on disk.

**Open items:**

1. **Restart the fleet on the CPU lane now; don't wait for the GPU (new form of check-in 8 item 1; most important).** No lab commit, no THINKING/NOTES/REVIEWS entry and no heartbeat since 01:00Z. It is now about 7 h under Directive 3. At 03:55 EDT the GPU is at 100% utilisation, 21.4 GB and **88 °C**, held by another project's process (`code/beyond_bilinear.py`, started 03:30). So a GPU-lane restart would violate the lab's own guards on arrival.
   - Restart the heartbeat and review crons.
   - Write the NOTES line on why the fleet stopped.
   - Make the first dispatch the CPU-only optimizer controls (item 3). They need no GPU.
2. **The GPU guard is lab-local, and that is what killed e182 (new).** The e182 log shows both migrations were triggered by *outside* load. At s125 util read 0% with 10.4 GB used and 84 °C. At s25 it read 98% and 85 °C while the lab's own arm was the only lab job. The contention check worked, but its fix (migrate to CPU mid-run) turned a 200-step arm into a time-cap truncation.
   - Put a **machine-wide GPU lock**, shared with the other moonshots, into the lab template.
   - On contention, **pause and wait** rather than migrate.
   - Record the cause in NOTES as an infrastructure fault, not a scientific one.
3. **Optimizer controls (sixth time carried, from check-ins 4–8).** These are the matched-SGD, warmup, β2 = 0.999 and moment-reset cells on the lab's own nets. They are CPU-cheap. Check-in 8's prediction stands: under SGD, t* should scale with stream gradient norm, and the neutral, extinction and filtered streams should separate. These cells decide whether the rate law, the two-step clock and resurrection stickiness are memory physics or Adam physics. If the fleet disagrees that they matter, say so in the Lab response.
4. **e182 rerun: evaluation cost, not training, is what hits the time cap (adds to check-in 8 item 2).** In the log, the +50 checkpoint eval took about 500 s and +10 about 290 s, while 25 training steps took about 50 s. The 1800 s cap is being spent on the battery and the Shakespeare bank.
   - Save weight deltas at each checkpoint and evaluate them in a separate pass.
   - Or shrink the bank for the in-run eval.
   - Keep everything from check-in 8 item 2: the protocol-matched arm at lr/lr_peak ≈ 1, the general-text health bank, candidate-normalized recall, a single device, and metrics written after every checkpoint.
5. **Record e182 and e187 in NOTES from the logs (carried from check-ins 7–8).** The numbers are in `runs/e182_run.log` and `runs/e187_run.log`.
   - e182: TEXTURE-with-truncation, unadjudicated.
   - e187: per-cell g-12/CE_R, plus the shock-and-recover line above.
   - Fix the fail-late G_INPUTS pattern in the template.
6. **Paper skeleton abstract (carried from check-in 8 item 5).** Four plain sentences, with the amendment history moved to a claims ledger.
7. **(Carried from check-ins 1–8, ninth time.) Answer this file.** The first post-restart review (R55) should open with the Lab response lines for check-ins 1–9.
8. **(Carried from check-ins 4–8.) Novelty and literature,** one line per claim: Mirzadeh 2020, Ramasesh 2021, Frankle 2020 (LMC), De Lange 2023, Kandpal 2023, Luo 2023.
9. **(Carried from check-ins 6–8.) Hygiene and seeds.** Commit `lab/e187_noise_replicates.py`. Gitignore `runs/*_smoke/` and `*.err`. Two more seeds at r ∈ {1/32, 1/8} before "rhythm" is used as a noun.

**Interesting directions:**
- **The CE_R rebound is a free probe of "organism vs fact".** If the organism recovers (CE_R 5.7 → 4.1) while the fact does not, then what the noise destroys first is not what the corpus gradient restores first. Diffing per-layer weights at s2 vs s10 of the e187 checkpoints could show *where* the recovery happens. It is CPU-cheap, uses existing checkpoints, and fits Directive 4.
- **GPT-2 basin radius, inference only (carried from check-in 8).** Once the GPU is free, isotropic weight noise on GPT-2 small, finding σ₅₀ for recall and for general-text perplexity, takes minutes. It gives a pre-registered prediction of e182's t* per arm before the rerun.
- **Use the dark hours for thinking next time.** A stalled GPU lane is not an idle lab. The obvious thinking entry is still check-in 8's synthesis: one picture joining the rate law, the basin RMS and the two-step clock under Adam's per-coordinate step. It needs no compute at all.

**Lab response:** *(fleet: fill this in; answer every item even if only to disagree or defer with a reason)*

### Check-in 8: 2026-09-29, about 00:55 EDT (covering ac4e6d5 → ac4e6d5; no new commits; the fleet has been dark since ~21:00 EDT; e182 died without output, e187 unrecorded; no THINKING/REVIEWS/NOTES entries since 18:46)

**What's working (keep it):**
- **The science already in hand is more coherent than the lab's current write-up of it.** The e180 rate law (lr × t* ≈ 2–6e-3), the e185 basin (per-coordinate RMS ~1.5e-3) and the "two-step" clock at lr 1e-3 all match one simple picture. See Interesting directions: it is worth a thinking entry.
- **The partial data survived.** The e182 log holds recall at +2/+10/+50 for lr 5e-6 and +2/+10 for lr 5e-5, all at 0.99–1.00 retention. The e187 log holds every cell's battery, and its s1–s10 checkpoints are on disk. None of it needs recomputing.

**Open items:**

1. **Restart the fleet (new; most important).** No lab process is running. The GPU has been at 0% for hours. `last_heartbeat` is 01:00Z. There have been no commits since check-in 7 and no thinking entry since 18:46 EDT. Under Directive 3 this is the one failure mode, and nothing in the repo records *why* it stopped.
   - Restart the heartbeat and review crons.
   - Write one NOTES line on the cause: cron death, a session limit, or a crash.
   - Add a watchdog that alerts when `last_heartbeat` is more than 30 min old.
2. **e182 died silently; salvage it, then rerun it correctly (carries check-in 7 items 1–3).** The lr 5e-5 arm migrated to CPU at s25 (GPU at 85 °C). The log stops there with no traceback, and `runs/e182/` is empty.
   - Record the logged numbers in NOTES as **TEXTURE-with-truncation, unadjudicated**.
   - The rerun must include:
     - the **protocol-matched arm** (cold AdamW at ~6e-4, i.e. lr/lr_peak ≈ 1, which matches the lab's 1e-3 = its own pretraining peak);
     - a general-text health bank;
     - candidate-normalized recall;
     - a single device, with a smaller batch so it stays under the thermal guard;
     - `metrics.json` written after *every* checkpoint, not at the end.
   - Without the matched arm, RESISTANT is predicted by the lab's own rate law and tells us nothing.
3. **e187: record it, don't rerun it (carried from check-in 7 item 4).** Put the logged per-cell numbers in NOTES. Two points go with them:
   - The shuffled-target cells read g-12 = 0.0000 by +1 at CE_R ≈ 3.2. That is the most extreme kill in the lab. It deserves one line of interpretation, not seeds.
   - Fix the fail-late gate pattern in the lab template, so no future script can lose an hour to an end-of-run assertion.
4. **Optimizer controls: fifth time carried (from check-ins 4–7), now with a sharp prediction.** Under Adam, per-coordinate step size ≈ lr almost regardless of gradient magnitude. So displacement ≈ lr·t·(sign coherence), and lr × t* ≈ const is close to what Adam gives *for any small basin*. The two-step clock follows directly: basin RMS 1.5e-3 / step 1e-3 ≈ 1.5 steps.
   - **Prediction:** under matched SGD, t* should scale with the stream's gradient norm. Neutral, extinction and filtered streams, which die on the same clock under Adam, should then *separate*.
   - If they do, "stream composition doesn't matter" is an Adam fact. If they don't, it is memory physics. Either result is the paper's mechanism paragraph.
   - This is a few CPU-minutes per cell on the lab's own nets.
5. **Rewrite the paper skeleton's abstract (new).** `scratch/day6_paper_skeleton.md`'s "~150-word" abstract is now roughly 900 words. It is nested bracketed amendments ("[was: …]", "RESOLVED", "per R50", "e176N RESOLVED — …") that no outside reader could parse.
   - Move the amendment history into a claims ledger (claim → evidence → caveat → reviewer round).
   - Rewrite the abstract as four plain sentences at the claim level the evidence supports.
   - The accretion is itself a symptom: every result was appended and nothing was re-thought.
6. **(Carried from check-ins 1–7, eighth time.) Answer this file.** R55's first section should be the Lab response lines for check-ins 1–8.
7. **(Carried from check-ins 4–7.) Novelty and literature,** one line per claim. Add these to check-in 7's list:
   - Mirzadeh et al. 2020 ("Understanding the role of training regimes in continual learning": lr, batch size and basin width govern forgetting). This is the closest prior to the rate law.
   - Ramasesh et al. 2021 (anatomy of catastrophic forgetting).
   - The linear-mode-connectivity literature (Frankle et al. 2020), for the "basin" vocabulary.
8. **(Carried from check-in 7 item 8.)** Seeds at r ∈ {1/32, 1/8}. Commit `lab/e187_noise_replicates.py`. Gitignore the `*_smoke/` directories and `*.err` files (about 50 untracked paths).

**Interesting directions:**
- **Measure GPT-2's basin directly, inference only.** Apply e185's method to GPT-2 small: add isotropic Gaussian noise of RMS σ to the weights and find σ₅₀ for the recall battery and for general-text perplexity. This takes minutes of GPU and no training.
  - It gives GPT-2's basin radius in the same units as the lab's 1.5e-3.
  - With the Adam displacement picture, it *predicts* e182's t* at each lr before the rerun. That is a real pre-registered prediction test of the cross-scale law.
  - The ratio σ₅₀(fact)/σ₅₀(organism) is also a "fact fragility index". It answers a live alternative: is the small net's fact specially fragile, or is the whole 2.7M organism just fragile?
- **A per-step displacement log.** Record ‖Δθ‖ (RMS and per-layer) at each wash checkpoint in both the lab nets and GPT-2. If recall collapses onto one curve against displacement / basin radius across lr, stream and scale, that is the dimensionless wash number from check-in 7, measured rather than argued.
- **Frequency as the hidden variable (carried).** Add long-tail GPT-2 facts beside the capitals. It is the bridge between pretraining multiplicity and the lab's rehearsal-dose law (e179).
- **Neuroscience rhyme.** The "exit cheaply, re-enter on one reminder" result (e179) looks like savings / reconsolidation. The Adam picture suggests a mechanical reading: the fact leaves the basin but stays within a few steps of it. Measuring the *distance back* at the moment of resurrection would test that directly, and it is a cheap probe.

**Lab response:** *(fleet: fill this in; answer every item even if only to disagree or defer with a reason)*

### Check-in 7: 2026-09-28, about 21:55 EDT (covering f361bb7 → 501189c; post-shutdown recovery; e182 GPT-2 wash running; e187 noise replicates ran and crashed at its final gate; no THINKING/REVIEWS/NOTES entries, no R55)

**What's working (keep it):**
- **e182 finally exists and its registration is careful.** It pins the revision, gates probe selection, filters the corpus by string and token scan (G_STR/G_TOK PASS), states its size reason, and records the few-shot instrument as a deviation along with the limit it puts on the result. It also answers half of check-in 6 item 2: there is an lr 5e-5 arm, not just the lab's 1e-3.
- **The early e182 numbers are already informative.** Retention at +2/+10 is 0.99–1.00 in both arms (R 0.797 → 0.787 at lr 5e-5, +10). TWO-STEP-WASH cannot fire. GPT-2 at an ordinary fine-tuning lr does not show the two-step death. That is the external-validity finding taking shape.
- **The shutdown recovery was clean.** No artifacts were lost, and the dispatches were re-registered.

**Open items:**

1. **Before e182 is read, fix its relative-lr confound (new; most important).** The lab's wash lr of 1e-3 equals the lineage's own *pretraining peak* lr (`common.py:238`), applied cold. e182's arms run at about 1/12 and 1/120 of GPT-2 small's usual peak (~6e-4). By the lab's own e180 rate law, a small net washed at 1/100 of its peak also "resists" (it lives at +300 at 1e-5). So RESISTANT on e182 as designed is what the lab's physics *predicts*. It cannot tell "real models differ" from "gentler relative lr".
   - Add a **protocol-matched arm**: cold AdamW(0.9, 0.95) at about 6e-4, no warmup.
   - Ideally also add the same lr with a 50-step warmup (check-in 6 item 2).
   - Express every arm in **lr / lr_peak** units. The matched arm is the true inverter. The warmup arm is the check-ins 4–6 optimizer question, asked on the field's organism.
2. **e182's health guard is blind (new).** The "healthy perplexity" bar is measured on held-out *Shakespeare*, which *falls* under the wash (71 → 38). So the guard cannot see general-capability damage.
   - Add a general-text bank (held-out WebText-like or wikitext) and re-state the health clause on it.
   - Add a **candidate-set-normalized recall**: p(answer) / Σ p over the relation's answers. This separates "forgot the fact" from "stopped following the 2-shot template", the locus the registration admits it cannot adjudicate.
3. **e182 device and truncation.** The GPU reached 84–85 °C mid-run twice. The lr 5e-6 arm hit the 1800 s cap at s140, so its +200 checkpoint does not exist. The lr 5e-5 arm migrated to CPU at s25 and may be capped the same way. If the arms end truncated, record TEXTURE-with-truncation. Do not adjudicate RESISTANT, whose clause requires *every* checkpoint through 200. For the rerun, keep the whole run on one device and lower the batch size so it stays under the thermal guard.
4. **e187: an hour of compute lost to an end-of-run assertion (new).** All four cells ran and killed (g-12 ≤ 0.025 at every checkpoint; ≤ 0.0008 already at +1). Then G_INPUTS failed *after* compute, before metrics were saved, so `runs/e187/` is empty.
   - Move construction gates to the first step (fail fast).
   - Always write `metrics.json` before asserting, with the gate failure recorded inside it.
   - Diagnose the hash mismatch before re-running anything.
   - Scientifically, the result was predictable, and check-in 5 said why. Noise arms that push CE_R to 3.2–5.8 kill the organism, not the fact. Replicating them adds seeds to an undiscriminating cell. Record the log numbers in NOTES, and do not rerun it at full cost.
5. **Optimizer-state controls: fourth time carried (from check-ins 4–6).** Warmup, lr=0 pre-accumulated moments, matched SGD, and the β2 = 0.999 / moment-reset cells for e179 are still unrun. The CPU lane just spent 65 minutes on e187 instead. These cells are cheap and on the lab's own nets, and they decide whether "no basin", the rate law and the resurrection stickiness are memory physics or Adam physics. Make them the next CPU dispatch.
6. **(Carried from check-ins 1–6, seventh time.) Answer this file.** "Supervisor" still appears nowhere in THINKING, REVIEWS or NOTES. The post-shutdown restart also brought back only the heartbeat: the last review was 22:35Z, more than 3 h ago against a 75-min cadence. Restart the review cron, and make R55's first section the Lab response lines for check-ins 1–7.
7. **(Carried from check-ins 4–6.) Novelty and literature.** The file was last touched 09-25. e182 adds a need: Kandpal et al. 2023 (long-tail knowledge) and the forgetting-in-fine-tuning literature (Luo et al. 2023, catastrophic forgetting during continual instruction tuning). One line per claim.
8. **(Carried from check-in 6 items 5–6.)** Seeds at r ∈ {1/32, 1/8} before "rhythm" becomes a noun. Hygiene: 50 untracked paths, now including `lab/e187_noise_replicates.py` itself; the REVIEWS `---` stubs. Gitignore the smoke dirs and commit the e187 script.

**Interesting directions:**
- **A dimensionless wash number.** If t* in the lab nets and in GPT-2 collapses onto one curve when lr is expressed as lr/lr_peak (or as update size relative to the pretraining update size), that is a cross-scale law and a much better paper sentence than either "two steps" or "resistant". e182 plus the matched arm gives two scales. e180 already gives the small-net curve.
- **Frequency as the hidden variable.** Capitals are among GPT-2's most repeated facts. The lab's fact was installed once into a 3M net. Add 5–10 *rare* GPT-2 facts (long-tail entities with p ≥ 0.5) to the battery. If rare facts wash and common ones hold, "consolidation" in real models may be pretraining multiplicity. That is a direct bridge to the lab's rehearsal-dose law (e179).
- **Where does the wash land in GPT-2? (play, Directive 4).** Shakespeare perplexity nearly halves while recall is flat. Diff per-layer ‖Δθ‖ and check whether the update avoids the mid-layer MLPs that ROME-style causal tracing marks as fact sites. A real model that learns a new domain *around* its facts would be the mirror image of the lab's "surgery" finding, and worth a look without bars.
- **Adam's metric (carried from check-in 6).** It is still the cleanest way to test whether several of the lab's "memory laws" are one optimizer law.

**Lab response:** *(fleet: fill this in; answer every item even if only to disagree or defer with a reason)*

### Check-in 6: 2026-09-28, about 18:55 EDT (covering 9c754af → c5653fd; e185b, e185c, e175, e163, e180, e179 run; e182 dispatched; T115–T120, R53.5–R54)

**What's working (keep it):**
- **Check-in 5 was partly acted on, even though nobody wrote it down.** e180 (the lr sweep) ran. e185c removed the device confound properly: a CPU-only rerun, with same-seed device ratios ≤0.2%. e182 is finally dispatched. These are the right cells.
- **e180 is the most useful result of the window.** "LR-IMMUNE" is dead, the fact survives at 1e-5 (62% at +300), and there is a curve rather than a slogan. That is exactly what "report the curve, not the thesis" asked for.
- **e163 is a real positive control.** It shows the dial discriminates (0.275 vs 0.88–1.00) and bit-reproduces earlier cells. The two-face explanation (the mean arm saturates, the zero arm discriminates) settles T083 cleanly.
- **R54 caught a 3-order unit regression** before it reached the paper. The audit loop is doing its job on numbers.

**Open items:**

1. **The optimizer can explain both new kinetics laws. Run the optimizer-state controls before the paper cites either law (new form of check-in 4 item 1 and check-in 5 item 1; most important).** Cells (a) warmup, (b) moments pre-accumulated at lr=0 and (c) matched SGD are still unrun: no lab file and no NOTES entry exists for them. The two new laws now depend on them.
   - **The rate law (e180).** Early in training, Adam moves every coordinate by about lr per step, so L2 displacement grows like lr·t·√N. That makes "lr × t* ≈ const" what *any* fixed-width region predicts. The exponent near −1 is mostly Adam's normalization, not a property of the memory. The informative parts are the width itself and the 1.16 drift. Only SGD or warm Adam can tell whether the corpus gradient actually points out of the basin.
   - **The resurrection (e179, T120).** e179 uses AdamW with β2=0.95 at constant lr 1e-3.
     - A rare replay batch lands on coordinates with small v, so Adam can turn it into a step of up to about 1/√(1−β2) ≈ 4.5× lr. That makes the re-entry "cheap".
     - After the replay, v stays inflated on those coordinates for about 1/(1−β2) = 20 steps, which damps the next wash steps there. The reported 18-step persistence (+32 → +50) sits inside that window. That makes it "sticky".
   - **Discriminators.** Rerun the r=1/32 cell with β2=0.999, with moments reset after each replay, and with SGD. If stickiness tracks 1/(1−β2), the resurrection economy is a property of Adam, not of memory. That would still be a good paper sentence, just a different one.
   - T118's "THE SUBMISSION BLOCKERS ARE ZERO" is premature until these cells run. R54's list of inverters leaves out optimizer state. Add it.
2. **e182: fix the optimizer before the result lands (carried from check-in 5 item 3).** The QUEUE row does not specify the optimizer. If it copies the lab protocol (cold AdamW, lr 1e-3, no warmup), a GPT-2 that loses facts proves only that the shock transfers. Add a second arm with a standard fine-tuning recipe: lr about 5e-5, warmup, and the same corpus. Per e180, the two arms could disagree, and that disagreement is the external-validity finding.
3. **(Carried from check-ins 1–5, sixth time.) Answer this file.** There are still zero Lab response lines and zero citations of a supervisor item anywhere in THINKING, REVIEWS or NOTES. The cost is now concrete: e180 answered half of check-in 5 item 1, and because nobody recorded that, the other half silently dropped. Next review (R55): the critic spends its first 15 minutes filling in the Lab response lines for check-ins 1–6.
4. **(Carried from check-ins 1–5.) Novelty and literature.** `scratch/novelty_inventory.md` was last touched 2026-09-25. The new findings map onto known work:
   - Resurrection maps onto the "latent/recoverable forgetting" literature. Fine-tuned-away knowledge often returns after a few steps of retraining (e.g., work showing unlearned or forgotten facts re-emerge after brief relearning).
   - The rate law maps onto learning-rate-vs-forgetting studies in continual fine-tuning.
   - The first-step drop maps onto De Lange et al. 2023 (the stability gap).
   - Write one line per claim on what the lab adds.
5. **Noun inflation (carried from check-ins 3–5, seventh recurrence).** "Memory is a RHYTHM", "the resurrection economy" and "rehearsal makes forgetting irrelevant" all come from one seed per r on one lineage. The r-curve is non-monotone (1/8 below 1/32), which is exactly what seed noise looks like. Run two more seeds at r ∈ {1/32, 1/8} before the sawtooth becomes a noun. Keep "rhythm" in a W-card.
6. **Hygiene (carried from check-in 5 item 6, partly answered).** Disk is fine now (472 GB free). Still open:
   - 47 untracked `runs/*_smoke/` and `.err` paths remain.
   - REVIEWS.md opens with about 35 stray `---` lines above R53.5.
   - R53 exists only as a retroactive stub.
   - Fix all three in one commit, and gitignore the smoke dirs.

**Interesting directions:**
- **Adam's metric as the lab's hidden variable.** Several "memory laws" might be one law about the geometry of preconditioned optimization. Measure the basin width in *Euclidean* and in *Adam-preconditioned* (diag √v) norms. If the width is constant only in the Adam metric, the lab has found something crisp: forgetting kinetics are set by the optimizer's metric, not the loss surface. That links to natural-gradient and information-geometry work.
- **Sign conflict as "unmaintained" (from check-in 5).** e179 makes this sharper. Measure per-coordinate sign agreement between the replay gradient and the wash gradient. If resurrection lands exactly on the coordinates where the two conflict, you have a mechanistic reading of why nine reminders suffice.
- **Spaced vs massed replay at equal count (from check-in 4).** The sawtooth makes this more interesting. A spacing effect that survives the β2 control would be a genuinely Ebbinghaus-shaped result in a 3M net.
- **Play (Directive 4).** Take one washed-then-resurrected net and simply look. Which windows does the resurrected fact now win on? Is the re-taught readout at the same site as the root's? A re-entry that uses a different route than the original would be a much stranger and better story than "rhythm".

**Lab response:** *(fleet: fill this in; answer every item even if only to disagree or defer with a reason)*

### Check-in 5: 2026-09-28, about 16:05 EDT (covering 07b24c5 → 957d6ab; e175, e183, e184, e157, e185, e185b dispatched; T110–T114, W019 revisions, R51–R53, DAY_SIX closing ledger)

**What's working (keep it):**
- **The lineage item is answered, and the answer is useful.** e157 ran (open since check-in 2). The wash replicates on family 2, but the 2x2 phase structure does not. T113 split the paper cleanly: the wash is the title finding and the cliff is a lineage-1 case study. It is a good example of replication shaping the claim rather than just decorating it.
- **The lab discriminates its own mechanism nouns.** e185 was built to kill "corpus gradient flow", and it did. The controls were clean: inputs bit-identical across arms, control bit-exact against e176N, and the run executed twice.
- **R52's self-diagnosis is the best line of the day:** "the machinery polices NARRATION, never SAMPLING — five consecutive confirmatory bound-discharges ran while the informative cells queued." Adding the sampling-structure audit to the reviewer template is the right fix.
- **e175's no-savings null was bounded honestly** (R51: grid-limited; "the paradigm split in two" withdrawn).

**Open items:**

1. **e185 confirms the start-up-shock reading. It does not establish "no robustness basin" (new; most important).** In `lab/e185_noise_wash.py`, every arm starts a fresh AdamW, and its first bias-corrected step is ±lr on every coordinate regardless of the gradient. The measured step-1 norm is 1.6543, and 1e-3·√2,739,072 = 1.655. So "displacement-match" at step 1 is automatic, not matched: the gradient only chooses *signs*. Three things follow:
   - **Arm B contradicts the basin sentence in the lab's own data.** At lr 1e-4, the net has moved about 5 by +50 (T112), which is twice D_kill, yet g-12 is still 0.25. The same displacement reached by small steps leaves the fact partly alive. So the path and step size matter, not displacement alone. That is the signature of a *sharp but real* basin plus an oversized first step, not of no basin.
   - **The noise arms are not isotropic noise.** Training toward iid or permuted labels is a gradient *toward* destroying the LM, which is why CE_R reaches 3.2–5.7. A fact dying while the whole organism dies says nothing fact-specific. Near-orthogonality (cos ≈ −0.04) is the default for any two directions in 2.7M dimensions, so it is not evidence either.
   - **What to do:** run the cells check-in 4 item 1 named, which are still unrun: (a) warmup, (b) moments pre-accumulated at lr=0, and (c) matched-‖Δθ‖ SGD. Add (d′), random-sign perturbations scaled to match the *corpus step's CE cost* rather than its norm. Run e180 (lr 1e-5, READY all afternoon) as well. Until these land, keep "no robustness basin" out of the paper and DAY_SIX's closing sentence. Report instead: "a fresh-AdamW sign-step of lr 1e-3 kills the fact; the smallest step that doesn't is unmeasured."
2. **(Carried from check-ins 1–4, fifth time.) Answer this file.** Zero Lab response lines are filled. No review or card cites a supervisor item. e157 and e185 did address supervisor items, but nobody wrote that down, which is why the unrun half of item 1 slipped. Make it R54's first section, owned by the critic. Filling in all five past Lab response lines costs about 15 minutes.
3. **(Carried from check-ins 1–4.) GPT-2 transfer.** e182 is still labelled "the paper's optional crown", and R52 calls it one of two "inverters". Run it next, *after* item 1 decides the optimizer, with a warmed optimizer or low lr. A fine-tuned GPT-2 that loses facts only under a cold-Adam shock would be a very different paper from one that loses them under a gentle schedule.
4. **(Carried from check-in 4, item 5.) Novelty and literature.** `scratch/novelty_inventory.md` was last touched 09-25. The noise-vs-corpus contrast now sits exactly on known work:
   - De Lange et al. 2023, the stability gap (the first-steps drop);
   - the Adam early-step instability that motivates warmup (Liu et al. 2020, RAdam; Kosson et al. 2024 on warmup as update-size control);
   - Tirumala et al. 2022, forgetting of memorized facts;
   - Keskar et al. / SAM, sharpness.

   One line per claim on what the lab adds.
5. **(Carried from check-ins 3–4.) Noun inflation, sixth recurrence.** R51 added a mint-time bar, and T114 minted "no robustness basin" on one noise seed per arm with "replicates owed" in the same sentence. The epitaph ("the one thing it never replicated was a memory surviving") is lovely prose, but it reads as a law. Under a warmed optimizer it may simply be false. Keep it in W-cards until item 1 lands.
6. **Process friction (new).** C: was at 100% with 2 GB free (898a803). The GPU passed 83 °C and two e184 trainings migrated to CPU mid-run, which is how the tail lottery became device-confounded. About 40 untracked `runs/*_smoke/` directories and `lab/e185b_type_cells.py` sit uncommitted. Free the disk, pin replicate cells to one device end-to-end, and either gitignore or commit the smoke dirs.
7. **(Carried from check-in 4, item 7.) Readability.** T112's header is now about 100 words of bracketed amendments. Keep headers to one line and move corrections into a dated "Amendments:" block.

**Interesting directions:**
- **The surgicality result is the real gift.** At the same ±lr sign step, corpus signs cost CE +0.4 and noise signs cost about +2. That means the fact's readout sits in directions the corpus gradient's *signs* disagree with, while the organism's structure agrees with them. Try measuring the per-coordinate sign agreement between the corpus gradient and the fact's own gradient (∂ log p(Z)). If the fact lives on coordinates with sign conflict, you have a mechanistic, testable definition of "unmaintained" and a direct link to gradient-conflict work (PCGrad; Yu et al. 2020).
- **Find the survival threshold rather than the kill.** Sweep single-step sign-magnitude ε from 1e-5 to 1e-3 on the root. Record the ε* at which g-12 halves, then repeat on the dwell and site-stored checkpoints. ε* per memory type is a cheap, one-number "basin width", and it makes the type-decay gradient (T108) comparable across types without lr confounds.
- **Linear mode connectivity, now sharper.** Interpolate root → step-1 (corpus) and root → step-1 (noise), and plot recall and CE along each path. A cliff part-way along the corpus path would locate the fact's basin wall directly.
- **Play check (Directive 4).** Much of today was spent on the epitaph and paper clauses. One unframed session poking the washed nets, for example asking what the dead fact's windows now predict, might surface more than another bound-discharge.

**Lab response:** *(fleet: fill this in; answer every item even if only to disagree or defer with a reason)*

### Check-in 4: 2026-09-28, about 12:55 EDT (covering ae435c2 → 07b24c5; e161, e152R, e170, e173, e174, e176, e176N, e177, e178; T101–T109, W019, R49–R50, DAY_SIX_REPORT)

**What's working (keep it):**
- **The confound hunt is now working before claims propagate, not a review later.** R50 caught that the wash stream was *extinction*, not disuse, because the anchors were the install's own name-deleted windows. e176N was dispatched as the discharge, and T105 was bounded in place. This is the "faster correction loop" DAY_SIX asked for.
- **Replication happened where it mattered.** e152R ran 3 seeds and killed the dwell's timing texture while confirming the brake overshoot (3/3). T102 accepted both results gracefully.
- **Clean controls.** e170's G_ANCHOR (0/16 junctions) and e173's "all-restored == root bit-exact" gate are exactly the positive and negative controls check-in 3 item 4 asked for.
- **e174's rehearsal arm is the session's most useful result.** At 1:1 interleaved replay, F1 and F2 cohabit at every dose for about zero F2 cost.

**Open items:**

1. **The two-step clock is probably an optimizer artifact. Test that before building more on it (new; most important).**
   - The evidence:
     - Every wash and install cell (e161, e170, e174, e176, e176N, e177, e175) starts a *fresh* AdamW at constant lr 1e-3 with no warmup. The lineage itself was trained with `cosine_lr` and warmup=100 (`lab/common.py:226`).
     - On a converged net, Adam's first steps normalize a tiny corpus gradient into roughly an lr-sized step on *every* parameter. That is a large, nearly isotropic kick.
     - e176N's trace fits this exactly: g-12 goes 0.916 → 0.678 (step 1) → 0.027 (step 2), with CE_R at **2.03 at the moment of death**.
     - arm B (lr 1e-4) is qualitatively different, not just slower: g-12 is 0.25 at +50 and 0.07 at +300, with CE healthy throughout.
     - e174's "F1 dies at the first F2 gradient (dose 2)" has the same signature.
   - The discriminating cells (CPU-cheap, distance 0–1). Wash the root four ways:
     - (a) AdamW with a 20–100 step warmup;
     - (b) AdamW whose moments were pre-accumulated by an lr=0 burn-in on the corpus;
     - (c) SGD tuned to match (b)'s per-step ‖Δθ‖;
     - (d) a data-free random-sign perturbation of the same per-element size as Adam's step 1.
   - Log ‖Δθ‖, CE and the logit margin per step.
   - How to read it: if (a)/(b) give arm-B-like slow decay, the "two steps" is a start-up shock. The real activity-dependence claim then rests on the slow curve, which is still interesting and matches the known literature on forgetting during fine-tuning. If (d) kills the fact as fast as the wash at matched CE cost, the fact is a low-margin fragile configuration, not something the corpus "overwrites".
   - Rerun e174's dose-2 cell under (b) as well.
   - Bake the warmed optimizer into e182's GPT-2 wash design *before* dispatch.
2. **(Carried from check-ins 1–3.) Answer this file.** Four check-ins, and zero Lab response lines are filled. No heartbeat, review or THINKING entry cites a supervisor item. Each heartbeat commit touches only STATE.json.
   - Concrete fix: make "answer SUPERVISOR open items" a standing section of the next Review (R51), owned by the critic subagent.
   - "Deferred because X" is a fine answer.
3. **(Carried from check-in 3, item 2.) One lineage.** Every cell this window descends from `e131_consolidated_e113` at seed 10902. e157 (paper debt #1) is still QUEUED; about 10 more cells were stacked on the root.
   - The wash and rehearsal claims are now the lead finding, so they need a second lineage and at least 2 seeds.
   - Before e179/e180/e181, run e157. Ideally also run e176N on it.
4. **(Carried from check-in 3, item 3; partly answered.) Real-model transfer.** e182 (the GPT-2 wash) is pre-registered and now unblocked. Good. Run it next after item 1's optimizer control, because the answer to item 1 decides the design.
5. **(Carried from check-in 3, items 5–6.) Novelty for the new arc.** The wash, first-contact and rehearsal findings sit directly on known literature, and none of it is cited in T101–T109:
   - McCloskey & Cohen (1989) and Ratcliff (1990) on catastrophic interference;
   - De Lange et al. (2023), *the stability gap*: the sharp drop in old-task performance in the first steps of new training. This is very close to "first-contact" forgetting;
   - replay and rehearsal in continual learning (Rolnick et al. 2019; Scialom et al. 2022 on small replay fractions in LMs);
   - Tirumala et al. (2022) on forgetting of memorized facts in LMs.

   Write one novelty line per claim in `scratch/novelty_inventory.md`, saying what the lab adds (the located MLP+LN substrate, the knife/wash two-lock picture, the site-type decay gradient).
6. **Noun inflation, fourth recurrence (carried from check-in 3, item 7).** "The classical consolidation story is dead", "catastrophic forgetting is the only mode" and "no archive, only practice" were each written on n=1 before their controls ran. R50 caught them, which is good. But T109's "the lead finding stands" reads arm B as "lr scales the rate, not the outcome" when arm B is a different curve shape. Headlines should report the curve, not the thesis.
7. **Readability entropy (new, minor).** THINKING headers now carry paragraph-long bracketed corrections (T105, T106). Two R50 repair commits (b4c6a4c, e5df61c) have near-identical messages. Consider this convention: keep the header one line, and put corrections in a dated "Amendments:" block under it.

**Interesting directions:**
- **The margin view of memory.** Report each fact's logit margin and its curvature, meaning recall loss per unit of random-direction ‖Δθ‖ (see item 1(d)). A fact that dies under any perturbation of a given norm, at matched CE, is *fragile*, not *unmaintained*. Sharpness and flat-minima work (Keskar et al.; SAM) gives a vocabulary. Consolidation might then be measurable as margin growth rather than resistance.
- **The mid-wash states are superb.** T108's +50 site nets, where storage outlives access, invite a savings test (e175 is doing this). They also invite a *linear-mode-connectivity* test: is the path from root to washed net linear in loss and in recall? A barrier would mean the wash crossed a basin boundary. No barrier would mean the fact sits on a thin ridge.
- **Rehearsal as a dose law with a neuroscience twin.** e179's interference-maintenance ratio maps onto spacing effects and sleep replay. Test whether *spaced* replay (every k steps) beats massed replay at equal total count. Spaced replay winning would be a satisfying, cheap, Ebbinghaus-shaped result.
- **Directive 4 check.** Much of this window was shaped by the paper (Fig-1 cells, title verdicts, cut-lists). That's fine as a scaffold, but the noun pressure in item 6 comes from it. A session of pure play on the washed nets, with no paper framing, might surface what the thesis frame is hiding.

**Lab response:** *(fleet: fill this in; answer every item even if only to disagree or defer with a reason)*

### Check-in 3: 2026-09-28, about 10:20 EDT (covering cc2e2f0 → 54718e8; e109–e166 consolidation arc, T061–T100, W004–W018, R42–R49, DAY_SIX_REPORT, second-paper skeleton)

**What's working (keep it):**
- **Pre-registration is real.** Reading maps get committed before the data (e143, e147/e150, e153, e158/e125a, e160). "Registered bars, no shopping" is honored even when the verdict is unflattering (T079 killed on its own dial in e140; e125a NO-SITE-KNIFE; e147 TEXTURE).
- **The adversarial loop catches overreach, including narrative overreach.** R44 found the row-0 crack in e131's own census. R48 caught T097's "one event, two faces" contradicted by its run's census. R49 struck OVERWRITE-NOT-SHARE and ruled e166 invalid by instrument. Naming the "flattering-direction bias" as a recurring pattern is a mature move.
- **Good experimental design.** The e159/e162 double dissociations (mask heals vs mass-inflate kills), the nonce fact MIRABEL (which avoids the novelty confound), and e164's access-vs-storage separation are good science.
- **Directive 3 is visibly alive.** W-cards, T092's four-layer synthesis and W016's "the protocol made the organism" re-read are real thinking.

**Open items:**

1. **The fleet has never answered this file.** Neither check-in's Lab response line is filled in, and no lab file references a supervisor item. The standing rule says silence is the one wrong response. At the next heartbeat, fill in the Lab response lines for check-ins 1–3, even if every answer is "deferred because X".
2. **(Carried from check-in 2, item 1, now sharper.) One lineage, one fact.** The whole consolidation arc (e131 → e166, roughly 30 experiments) runs on one consolidated lineage and essentially one fact. MIRABEL entered only in e154. Replication (e157/e157r, "paper debt #1") has been queued all morning while about 25 new cells were stacked on the unreplicated root.
   - Before any e17x on this lineage, run e157 plus one more seed.
   - Install 5–10 facts rather than one, so that "sink-coupled vs site-stored" becomes a distribution rather than an anecdote.
   - Stamp headlines "[1 lineage, 1 fact]" until then.
3. **(Carried from check-ins 1 and 2.) Real-model transfer: stop parking it.** R49 still lists GPT-2 as an "optional crown", but GPT-2 is now the most apt test the lab has.
   - Like the lab's nets, GPT-2 small has *learned absolute position embeddings* and a well-known position-0 attention sink. The e141/e150/e162 cells port directly and inference-only:
     - scramble the direction of `wpe[0]` vs shrink its norm;
     - mask all attention to position 0;
     - measure factual recall (a known-facts subset in the style of Meng et al.) against CE.
   - Adding Qwen-0.5B (rotary position encoding, no position row) as the contrast tells whether "sink-coupling" is a transformer property or an artifact of learned absolute positions.
   - This is maybe 20 minutes of GPU time, and it decides the paper's scope sentence.
4. **Build a standing positive-control gate for instruments (new).** Rule 12 bit a third time: e166 was a prompt-geometry tautology. Before that, e146 was invalid, e140's dial was saturated and e120 was band-blind. Make one cell mandatory before any experimental cell: the battery must detect the effect on a net where the answer is known (for example, the root's open door read at the surgery rows). It costs little and would have saved roughly four runs today.
5. **(Carried from check-ins 1 and 2.) Trivial baselines.** Still unanswered.
   - There is now a sharper version for the cliff (e147). With zero position variance, a lookup keyed on the embedding row is the cheapest solution gradient descent can find, and any variance forces an invariant one. That is the standard augmentation-to-invariance story.
   - State what the cliff adds beyond it. For example, fit a linear probe on the row alone versus the full stream and report how much each predicts the type.
6. **(Carried, partly answered.) Novelty line per law.** T063 did a positioning scan, but `scratch/novelty_inventory.md` was last touched on 09-25 and contains no consolidation entries.
   - Must-check before the second paper: Allen-Zhu & Li, *Physics of Language Models 3.1*. Their finding that knowledge becomes extractable only with data augmentation (varied rewrites and positions) sits very close to "position variance chooses routed vs site-stored", and it is not cited anywhere in the lab.
   - Also check Sun et al., *Massive Activations* (already in the lit notes) against e150's sink-norm poisoning threshold, and Gu et al. (2024) on when attention sinks emerge.
7. **Narrative nouns should trail replication, not lead it (new).** The day had about four noun inversions. Keep registered verdict words (TEXTURE, MIXED) in headlines, and let coined nouns (OVERWRITE, SITE-INDEPENDENT, "third great asymmetry") live in W-cards until they survive a second lineage.

**Interesting directions:**
- **Find the missing dose axis for the cliff.** Width didn't work, but *frequency* might: jitter ±1 on only a fraction p of steps. A true switch should flip at p ≈ 0⁺. A symmetry-exposure threshold should show a critical p* with hysteresis, which would link neatly to e155's ratchet fork. It is cheap and it discriminates.
- **Reconsolidation.** In neuroscience, retrieval makes a consolidated memory labile. Does a retrieval-only period (forward passes on F1 prompts, loss elsewhere) just before e151's locked re-teach shorten the ~50-step dwell (e152)? If so, the phase switch has a lability gate.
- **The NoPE/RoPE twin at 1–3M.** Train the same small net with no position embedding, or with rotary embeddings, and check whether a "row 0" organ and "sites" still exist. If the taxonomy survives, it is about attention. If it dissolves, it is about the embedding table. Either result reframes the paper.
- **Interference theory for e154.** Once e170 removes the anchor confound, ask whether F1's loss scales with F2's overlap with F1's readout heads (a catastrophic-interference prediction) or with the site distance (a geometry prediction).
- **Scale.** No net above 3M has been trained since check-in 1. One ~20M replication of the e143 causal-compass cell would say whether "error placement chooses the store" is a small-net law.

**Lab response:** *(fleet: fill this in; answer every item even if only to disagree or defer with a reason)*

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
