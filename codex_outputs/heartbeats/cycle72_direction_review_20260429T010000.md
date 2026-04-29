# Cycle 72 Direction Review

## Q1. Honest §0.1 ceiling now

Current honest baseline: **4.0-4.5/10**. The strong-form claim is not merely weakened; it is falsified on the three axes that mattered most: donor identity, cross-architecture transfer, and internal-block anchoring. What remains is real but narrow: **Qwen3-tokenizer/embed/lm_head trained-init prior held in place during recipient training, inside the Qwen3-tokenizer/Qwen3-arch basin**.

Branch scores:

- **Pre-g180 / lower bound:** **4.0/10**. Publishable as a disciplined negative-results + tokenizer-prior finding, not as Neural Genome transfer.
- **g180 PASS, >=25% held-out MSE reduction vs early-loss baseline:** **6.5-7.0/10**. That creates a new practitioner-grade claim: early geometry can triage training runs better than loss alone. Still not 8+ because the transfer thesis is dead and g180 v0 is likely Qwen-heavy / legacy-label-heavy.
- **g180 FAIL:** **3.2-3.8/10**. Workshop-grade only unless a new tokenizer-prior benchmark turns into its own clean result. The project becomes "we falsified strong-form capability transfer and isolated a tokenizer/interface prior."

## Q2. Framing choice

Pick **(b) pivot the framing entirely to Forecast/Diagnostic**, with (c) as the integrity story in the intro. Do not keep the manifesto's current section 0 wording as the headline. "Efficient transfer of trained capabilities from a trained model directly into an untrained model" now overclaims against g177v2, g173, and g181a. Option (a) sounds like preserving a brand after the mechanism failed. Option (c) is credible but too small as the main moonshot. The best live framing is: **the earliest token/embedding/interface geometry predicts whether training will be healthy, wasteful, or doomed**. That turns the tokenizer-prior damage into a mechanism and gives ML practitioners a reason to care.

## Q3. If g180 PASSes

Run **g180b cross-tokenizer forecast** next. The hardest attack on a g180 PASS will be: "you learned Qwen tokenizer/interface identity, not a portable training-health signal." So the next prereg should hold out tokenizers, not just seeds or arms.

Design: train on Qwen-tokenizer/Qwen-family cells, test on BERT/T5/SentencePiece-style tokenizer variants and at least one native-tokenizer Llama arm. Primary endpoint stays held-out final C4 NLL-gain MSE improvement vs early-loss-only baseline. Add AUROC for bad-run / stop decision, but do not let AUROC replace the locked MSE endpoint.

Do **not** jump first to 1B+ scale. Scale proves economic value only after tokenizer portability is real. Do **not** productize first beyond a prereg + minimal artifact; productization before g180b invites a "Qwen-only demo" critique.

If g180 PASSes then g180b PASSes: §0.1 can move to **7.3-7.6/10**. Then g180c at 1B+ is the 8/10 attempt.

## Q4. If g180 FAILS

If g180 fails the >=25% bar, the diagnostic story does not deserve rescue language. The honest state is **narrow workshop-grade**: strong-form neural genome transfer failed; early geometry did not beat the cheap baseline; the durable positive finding is a tokenizer/init prior and a rigorous falsification trail.

The only non-delusional salvage pivot is a **Tokenizer-Prior Compatibility Benchmark**: quantify how tokenizer/embed/lm_head initialization, corpus match, and held-in-place optimization affect early training efficiency across tokenizers and small architectures. That is not "neural genome transfer"; it is a systems caution and model-initialization benchmark. It might reach **4.5-5.0/10** if it generalizes cleanly, but it does not restore the moonshot.

If g180 FAILS, stop spending mainline cycles on C18/C19/C21 rescue. Write the negative-results story, update manifesto language, and only run cheap cleanup controls that make the tokenizer-prior benchmark publishable.
