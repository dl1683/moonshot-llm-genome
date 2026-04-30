I’m treating §0.1 as **3.2/10 current**. g188 says: static bridges are dead, exact lexical row identity is live. Scratch mean NLL 5.828; flow_bridge **-0.119**, char **-0.041**, direct_string **+0.478**, shuffled **-0.715**, random **-0.898**.

| Rank | Candidate | Obvious? | §0.1 Upside |
|---|---|---:|---:|
| 1 | **Token-row compiler**: train a small function from token bytes/context/frequency to Qwen-format embedding rows using the 84% exact matches as supervision; hold out matched rows, generate unmatched rows, then train. | No | **+2.4** |
| 2 | **Loss-guided unbalanced matching**: replace Sinkhorn with decoder-loss-selected sparse matching with a dustbin/reject option. Wrong rows are toxic, so the win is selective refusal. | No | +2.0 |
| 3 | **Pulse-anchor basin test**: apply direct_string anchor only for first 250/500/1000 steps, then release. Tests whether the signal is early basin steering rather than permanent crutch. | No | +1.7 |
| 4 | **g191 string-match decomposition**: matched rows vs unmatched/frequency/shuffle. Necessary, but mainly adjudicates the side finding. | Yes | +1.2 |
| 5 | **28-layer + overlap ladder**: GPT-2 full-depth, then lower-overlap tokenizers. Tests persistence and overlap scaling. | Yes | +1.1 |

**Pick: token-row compiler.** It is the highest-leverage because it turns the “84% exact overlap” critique from a weakness into supervision. g191 can prove exact shared rows matter, but that still leaves you with row reuse. A compiler can prove a stronger law: trained tokenizer geometry is learnable from token form plus corpus role, and can generate useful rows for tokens never exactly shared.

Clean PASS bar: held-out exact-match row prediction works, unmatched-token eval improves, and full training beats scratch by at least +0.30 nats without relying on copied target rows. That would move this from “shared-vocab trick” toward an actual tokenizer-interface construction method.

