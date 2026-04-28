Trajectory: **7.9/10**, with a real path upward but only if you stop spending marginal compute on “third mechanism” variants and start cashing out the law.

- **Q1. What gets this to >=8.5?** Not cross-architecture replication by itself. A cross-arch row is credibility, not a flagship. The upgrade path is: **end-task transfer on a meaningful benchmark**, ideally with an efficiency headline, plus a clean theory statement. The strongest version is still `g155`-style energy cash-out. A derivation is valuable, but by itself it probably moves this from “strong mechanism paper” to “important theory note,” not flagship, unless it predicts and wins a new intervention. The best framing is no longer “continuous anchor law”; it is **continuous donor-information-in-the-loss law**: persistent transfer appears only when donor information remains present throughout SGD.

- **Q2. Is there a cheaper cash-out than wall-power?** Yes: **capability-per-FLOP** or **teacher-equivalent capability at lower train+inference compute**. That is the serious fallback. If `g167`-style KD can move a much smaller student to meaningful C3 performance on HellaSwag/PIQA/Winogrande, and you account honestly for teacher-query cost, you have a publishable efficiency claim even without kJ. It is weaker than energy, because FLOPs are easier for critics to discount, but it is much better than “C4 NLL only.” I would score a strong FLOP cash-out around **8.1-8.4** if the ratio is large.

- **Q3. Is `g171` still worth running?** Low priority. After `C18 + C19 + R9 + R9b`, the core law is already well supported: **continuous constraint survives; zero-step and decay wash out**. `g171` only matters if `g170`/`g172` produce an ambiguity that routing can resolve. Otherwise it is likely incremental, not trajectory-changing. Competitive answer: shift compute to a new axis, not another nearby probe.

- **Q4. Narrative going forward?**  
  `g158c` identifies **where** donor information matters: transport-demanded positions.  
  `g165` shows one way donor information persists: **continuous weight-space constraint**.  
  `g167` shows a second independent way: **continuous output-space constraint**.  
  `g168` and `g169` kill the seductive false stories: **alignment alone** and **temporary scaffolding** do not transfer capability.  
  Headline: **capability is not injected once; it is maintained by the training objective over time.**

**One concrete change:** after `g170` and `g172`, **drop `g171` and replace the next major experiment with a preregistered cross-architecture, end-task KD cash-out scored in FLOPs**. Example: Qwen teacher -> Llama-arch student on `C3_macro`, reporting capability retention and total train+inference compute. That single move tests generality, produces a real cash-out, and keeps the door open for the wall-power version later.