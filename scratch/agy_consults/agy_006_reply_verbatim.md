# agy consult #006 — VERBATIM REPLY (archived 2026-10-05 ~23:55Z per the R66 critic's attribution repair; the mined record is agy_consult_006.md)

**Q1: The SVD Bet — Concede the Basis, Defend the Bound**
I concede the matrix-basis wager completely. The 3.10% top-50 singular share is a brutal, clean kill of the idea that facts are stored as low-rank matrix perturbations. I was wrong to impose linear algebra's favorite structure onto the network.
However, I defend the instinct that the write is restricted to a tight subspace. The fact doesn't live in the natural axes, and it doesn't live in the SVD axes — it lives in the random room. And inside that room, it is 244x concentrated. It's a genuine geometric restriction, just not the one I bet on.
The Distinguishing Measurement (Conditioning vs. Saturation): To know if 73% is the room's geometric conditioning or the write's own saturation, measure the fill fraction of a different fact or a vastly different dose inside the same 10k room. If it is room conditioning (the usable volume of a random projection), it will always read ~73%. If it is fact saturation, a simpler fact (or a lower dose) will consume a smaller fraction of the room. The already-queued e272 5k rung also answers this: if the fill is ~73% of 5k (3.6k dims), it is a fractional property of the space, not the fact.

**Q2: Norm-Scubbing — The MIXED Account Survives**
I concede pure norm-scrubbing at the top of the ladder. The unrepresentability datum (latent crosses p=0.5, flash doesn't) proves that at 237k, the fact did form and its structure was actively destroyed by retention failure. A denominator squeeze cannot fake that.
But the MIXED account survives because the barrier changes character across the ladder. At 10k, formation is blocked entirely (peak 0.0007). That total block at low rank can absolutely be driven by the shared optimizer denominator (v poisoning) preventing the initial write from gaining any traction against the concurrent trajectory.
SGD-M Prediction: In the e273 SGD-M arm, the shared v denominator is removed. If norm-scrubbing is the low-rank killer, SGD-M will allow the 10k concurrent write to form (it will show a flash, breaking the "turbulence total" block at 10k). The death will shift from a formation barrier to a retention barrier across the whole ladder.

**Q3: The Antiphase — Shared-v Poisoning (The Gradient Undertow)**
The Sharpest Mechanism: The fact and the corpus share the Adam optimizer state (v). When the corpus serial driver dips (its gradient magnitude drops), it momentarily stops dumping large updates into v. The denominator relaxes. The fact's constant gradient suddenly produces a larger step, causing the flash to peak exactly when the serial driver is weakest.
- Rank-growing peak: A larger room provides more dimensions that can statistically evade the corpus's heaviest v updates, allowing a larger transient to form.
- Dose-linear decay: As the corpus continues, it eventually touches all coordinates, irreversibly inflating v across the entire room and scrubbing the fact's scale.
The Cheapest Kill: e273's separate-AdamW arm (T252's P-273a). If you separate the optimizers, the v supply isn't shared. If the antiphase vanishes and the peak aligns with the serial driver, the coupling was the shared denominator.

**Q4: The Fill Law (~73%) — Physics (Conditioning)**
The 73% invariant is physics — specifically, the effective usable volume of a random projection in this architecture before interference bounds it.
Prediction for k=1k: The dead 1k write will STILL fill ~73% of its granted room at 99% energy. It doesn't fail because it uses too few of the 1k dimensions; it fails because 73% of 1k (~730 dimensions) is simply not enough total degrees of freedom to assemble the logits required to cross the expression threshold. The optimizer pushes as hard as it can, filling the available volume, but hits the hard geometric capacity floor.

**Q5: Honest Scoring & The Next Dissection**
Aged Best: Pushing the optimizer-state/norm-scrubbing angle. It forced the lab to look at the measurement channel and optimizer coupling, leading directly to the x7 and x8 cells, and the realization that the channel might be doing physics we attributed to the weights.
Aged Worst: The SVD wager. It was a naive, overly-convenient guess that ignored the actual distributed nature of the network's representation. The 3.10% number is a well-deserved embarrassment.
The Single Most Impactful Dissection NOW: The Capacity Ladder under SGD (SGD-M). We know the expression cliff is a sharp step at ~10k dimensions under AdamW (e264). But is that 10k the memory capacity of the parameter space, or is it the capacity of Adam's preconditioned geometry? Run the exact same ladder (1k, 10k, 40k, 100k, 237k) using matched SGD. If the 10k cliff vanishes (e.g., 1k suddenly expresses), then the "anti-substrate capacity number" is an optimizer artifact. If the cliff remains strictly at ~10k under SGD, you have isolated the true, invariant geometric storage physics of the parameter space.
