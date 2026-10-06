# Read-Only Consult #007 — The Second Pushback Round

**Q1: Concede shared-v relaxation?**
I concede it entirely. The separate-AdamW arm dying *harder* (0.0076x vs 0.0252x) is the kill shot. If the shared denominator was the weapon, unsharing it should have spared the write. Instead, unsharing worsened it. The variant that predicts this is exactly plain trajectory collision with uncoordinated normalizers. Two Adams, blind to each other's momentum and variance, taking preconditioned steps over the exact same parameters, actively shredding the 73.5% dense room occupancy. A shared-v at least forces a common metric; separate-v means they fight in different metrics, creating worse turbulence. There is no need to dress this up as a new "variant." It is parameter-space collision.

**Q2: e280 (SGD-M) and the sign-step law**
The capacity ladder question is absolutely alive, and e280 should run at the stable-but-undermatched lr. e272 just gave us the license: dose does not buy expression (the kept-matched 1k arm died). If dose doesn't buy expression, we don't need matched dose to test the formation floor. The question e280 answers is whether the ~1-2k dimensional floor is an artifact of Adam's preconditioned geometry (the 80% in-room sign step) or an intrinsic property of the parameter space. If SGD-M at *any* survivable learning rate shows the same 1k-2k floor, the capacity number is space-intrinsic. If 1k expresses under SGD-M, Adam created the cliff. Run it, accept the dose mismatch, disclose it, and find the floor.

**Q3: Ranking the queued dissections under the two-body verdict**
1. **The noise-corpus isotope (e278)**: The ultimate discriminator for trajectory two-body. If the kill is pure parameter-space turbulence, a max-entropy corpus should wreck the write exactly as hard as the semantic corpus. If it spares the write, the collision requires semantic alignment. This isolates kinetic energy from semantics.
2. **K-interleave ladder**: Tests the kinetics of the collision and the retention ceiling. If we give the write quiet time to form, can it outrun the collision?
3. **Orthogonal implant**: We know the room is 73.5% densely filled. Forcing an orthogonal implant tests if we can engineer around the collision in the remaining 26.5% space. 
4. **Instrumented span-decomposition re-run**: Dead. The antiphase was an artifact; its payload is moot.

**Q4: Scoring myself honestly**
- **#005 calls**: I took a bath on the SVD wager (3.1% vs my 99% bet). I was wrong about sparsity. But I was right about the counterfeit's active unraveling—the disguise took, but the slide was optimizer-driven.
- **#006 defenses**: I defended shared-v relaxation, and it got humiliated by the separate-AdamW test. I defended the antiphase, and it flipped sign on a baseline change. 
I have been consistently wrong whenever I bet on static geometric properties or single-channel bottlenecks, and right when I bet on the optimizer actively fighting the fact. I take my lumps. The instrument is always right.

**Q5: The most impactful dissection NOW**
Given the edge at (1k, 2k], the 73.5% dense fill law, the gap-rent wash, and the two-body parameter collision:

**1. The Subspace-Adversarial Corpus (The Guided Missile)**
Instead of a max-entropy noise corpus, construct a corpus whose gradients are explicitly projected to be *entirely orthogonal* to the 1k-2k room. If the corpus is forced to take steps strictly outside the granted room, does the concurrent write survive? If it does, the collision is strictly spatial overlap in the parameters. If it still dies, the optimizer's undertow is pulling the write out of the room regardless of where the corpus steps.

**2. The Fully-Formed Concurrent Test**
We know concurrent *formation* dies at 10k, and concurrent *retention* dies at 237k. What happens if we take a perfectly quiet-formed 10k fact (which we know expresses well) and *then* subject it to 1:1 concurrent corpus interleave? Does the established 10k fact die as fast as the 237k fact? This isolates the retention barrier from the formation barrier at the exact threshold where formation is possible.
