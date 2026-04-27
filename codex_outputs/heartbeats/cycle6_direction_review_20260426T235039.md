Current state is: `6/10` stands on `g156 PASS` plus `g152` attenuation; `g157/g157b` killed the only internal-mechanism route, so the remaining upside comes from sharper empirical control, cross-class extension, and product cash-out.

- `(a) continue as-is: g158 -> g159 -> g160 -> g161)`  
  Score uplift: `+1 to +2` if the front half lands, so roughly `7-8/10`.  
  Problem: `g161` is now low-leverage and operationally bad. It does not raise the ceiling beyond `7-8/10`, and the implementation note says the pure-PyTorch RWKV scan implies roughly `556 hr`, so this tail is not credible as an immediate next step.

- `(b) pivot early: g160 -> g155)`  
  Score uplift: `+0.5 to +1.5`, roughly `6.5-7.5/10`.  
  Upside: strongest manifesto cash-out if it works.  
  Cost: you skip the two cheapest remaining theory-shaping tests, so a positive `g160/g155` reads more like a Llama-local engineering win than a design law.

- `(c) third action: keep the front half, cut the dead tail)`  
  Score uplift: `+1 to +2`, still `7-8/10`, but with better expected value.  
  Sequence: `g158` finish, then `g159`, then `g160`; defer `g161` until a fused RWKV kernel exists.

Recommend `(c)`.

One concrete change: rewrite the post-`g156` queue now to `g158 -> g159 -> g160`, and mark `g161` as blocked/archived pending fused RWKV kernel support.

Why this is the right move:
- `g158` is already running and is the cheapest remaining sharp prediction test.
- `g159` is the cheapest way to prevent the whole story collapsing into “one Llama-family quirk.”
- `g160` is still informative even under weakened theory and is the bridge to `g155`.
- `g161` currently burns time without moving the score ceiling.

Decision rule after that:
- If either `g158` or `g159` kills, pivot immediately to `g160 -> g155`.
- If both pass or partial-pass, run `g160` with much stronger framing.