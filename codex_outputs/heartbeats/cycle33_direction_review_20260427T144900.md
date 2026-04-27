`g165` is still the right next move. `g158c` PASS makes `g162` the best closure move on the architecture-prior branch, but it does not overtake `g165`, because the project’s own lock now says the architecture-prior line is feeder/cash-out while the §0 transfer line is primary, and `g165` is the best runnable test on that axis right now ([CLAUDE.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:9>), [WIKI.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:96>), [WIKI.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:100>), [post_g158c_decision_tree.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/programs/post_g158c_decision_tree.md:8>)). So Q1: ordering changes only slightly in `g162`’s favor, not enough to beat `g165`. `g165` stays about `7.3/10`; `g162` stays `6.8/10`.

Q2: `C14 + C17` is enough for a major canonical theory finding now: two independent control axes locked in one family. It is not enough for a moonshot flagship claim. The repo is explicit that `g158c PASS` gets the branch to about `7.2/10`, while only `g155` breaks the §0.1 ceiling at `8.2/10` because it is the electricity-grade cash-out ([CLAUDE.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:50>), [WIKI.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:83>), [CLAIM_EVIDENCE_MAP.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:52>)). Honest framing: strong theory result, not manifesto-grade yet.

Q3: ROI on the §0 axis is still positive for finishing `g165`. It is not clearly positive for auto-committing `g166` after a FAIL. `g166` is draft-only, higher-envelope, and its own prereg says the required donor optimizer state may not even exist yet ([genome_166_optimizer_state_decay_anchor_2026-04-27.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_166_optimizer_state_decay_anchor_2026-04-27.md:83>)).

Queue score:
- `g165`: `7.3`
- `g162`: `6.8`
- `g166`: `~6.4` by my estimate
- `g155`: `8.2`, but blocked

Overall active queue: `6.9/10`.

One concrete change: if `g165` FAILs, do not auto-run `g166`; move `g162` ahead of it, and treat wall-power-meter procurement as the real parallel priority.