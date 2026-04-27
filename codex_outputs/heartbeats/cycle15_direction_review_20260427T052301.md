Recommendation: **(a) finish `g158` -> launch `g160`**. Do **not** spend the next cycle on `g159b`.

Why:
`g158` is the last remaining sharp theory discriminator. `g159b` is a salvage experiment for a scale-limited assay; at best it upgrades a supportive null into cleaner cross-class phenomenology. That is exactly the kind of “one more figure / tighter error bar” move your own §0.1 says to de-prioritize. `g160`, by contrast, is the manifesto cash-out: if it passes, the theory becomes a design rule with product consequences.

Estimated score uplift from current `~6.5/10`:
- **(a) `g158 -> g160`**: expected **+0.4 to +0.8**
  - If `g158` PASSes and `g160` PASSes cleanly: likely lands around **7.1–7.4**
  - If `g158` is weak/KILL but `g160` PASSes: still can reach **~6.9–7.2** as an engineering result
- **(b) `g158 -> g159b -> g160`**: expected **+0.5 to +0.9**, but with worse expected value per unit time
  - `g159b`’s incremental contribution over path (a) is only about **+0.1 to +0.3** if it rescues class extension
  - If it fails, you burn a cycle for almost no score movement
- **(c) something else**: expected **0 to +0.3** next-cycle uplift unless the “something else” is a brand-new mechanism route, which is not ready and is higher-risk than the already-implemented `g160`

Concrete sequencing decision:
1. Finish `g158`
2. Read verdict once
3. **Immediately launch `g160` as the next GPU job**
4. Leave `g159b` archived unless later you specifically need a bounded rescue for class-extension in a writeup

Competitive view: after `g157/g157b` died, the path to distinctiveness is no longer “more internal evidence.” It is either `g158` preserving some theory structure, or `g160` proving the idea selects a better cheap model. `g159b` is lower-leverage than both.