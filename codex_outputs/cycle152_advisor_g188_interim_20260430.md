I would not promote `char_overlap_no_ot` yet. Flow was also positive at step 500: mean `+0.057`, then decayed through zero by ~2k-2.5k and ended `-0.119`. Char’s `+0.146` is materially stronger, so it is a live signal, but the test is: still positive at 1500, clearly positive at 2500, and 3/3 seeds at 5000.

Updated §0.1 score: **3.2/10 interim**. Static OT bridge failing lowers the post-g183 `3.5`. If char-overlap finishes positive `>= +0.12` in 3/3 seeds, I’d raise to **4.1-4.4**, not 6+, because that would support tokenizer-overlap priors, not general architecture-compatible geometry.

Highest leverage: **finish g188 first**. If char survives, do **g188b/g191 char-overlap decomposition**: `char_init_only`, `char_anchor_only`, row-shuffled char map, frequency-bucket shuffled char map, direct string, and random same-degree controls. If char degrades, go to **g190 decoder-conditioned relearning**.

Why OT likely hurts: Sinkhorn is imposing the wrong conservation law. Tokenizers are not two balanced semantic distributions. They are asymmetric, frequency-skewed, hierarchical codebooks. Raw char overlap preserves local span identity and token-boundary evidence; Sinkhorn redistributes mass, smooths/smears source rows, and forces rare/irrelevant tokens into the plan. Since both arms use the same training/anchor machinery, the difference points directly at the balancing step.

Ratings for §0.1 breakthrough potential:

| Option | Score |
|---|---:|
| `g190` decoder-conditioned relearning | **7/10** |
| `g189` C23 content-causality controls | **5.5/10** |
| `g187` ultrametric diagnostic | **3/10** |
| `d` char-overlap decomposition if signal survives | **7.5/10** |
| `d` if char later degrades | **4/10** |

My call: **do not start g189/g187 next unless you need integrity/background work while GPU is otherwise free. The real fork is char survives -> decompose char; char dies -> g190.**

