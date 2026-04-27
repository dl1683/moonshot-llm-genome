**Q1**

`g165` is still the right next GPU slot, and I would **not** fold optimizer-state into it yet.

Reason: the g137 “positive final outlier” is being over-read. In the original record, the clean optimizer-state test is `resume_true` vs `resume_reset`, and that is **early-only** then gone by step 2000-4000: same CtQ, same final NLL, `resume_foreign` indistinguishable, `state_only` catastrophic ([WIKI.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:901>)). The meta-audit’s `+0.08` endpoint comes from comparing `resume_true` to `state_only`, because g137 is extracted as donor=`resume_true`, scratch=`state_only` ([analysis_early_help_meta_audit.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/analysis_early_help_meta_audit.py:169>)). That is not clean evidence that optimizer-state transfer persists on its own.

So the higher-leverage single question remains: **can decaying anchor rescue the universal early-help/washout law?** Keep `g165` clean ([post_g158c_decision_tree.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/programs/post_g158c_decision_tree.md:8>)). If it FAILs, then `g166 = optimizer-state + decay-anchor` is the right next factorial.

**Q2**

Yes: run the 30-minute CPU g137 follow-up **now**, in parallel with `g158c`. But treat it as **schedule calibration**, not as a reason to reformulate `g165`.

The exact isolated decay already looks monotone, not “late rebound”:
- step 1064: `resume_reset - resume_true = +0.0456`
- 1128: `+0.0448`
- 1512: `+0.0156`
- 2000: `+0.0061`
- 4000: `-0.0004`

So the audit should extract per-step `true-reset`, `true-foreign`, and `true-state_only` deltas with CIs and a half-life. Useful for `g166`; not strong enough to change `g165`.

**Q3**

If `g158c PASS_canonical`: finish architecture-prior only as **closure**. Promote the claim, write the lock note, keep `g162` staged, then retire architecture-prior as the **discovery** branch.

If `g158c PILOT_FRAGILE`: yes, that materially strengthens the §0 pivot. Then architecture-prior is feeder/cash-out only, not frontier.

**Queue score**

- `g165`: **7.3**
- Path A / `g162`: **6.8**
- Path B / `g158e`: **6.4**
- Path C / `g155`: **8.2**, but hardware-blocked

Overall actionable queue: **7.0/10**. No slot-order change. The only operational change I’d make is the parallel CPU g137 decay audit before `g165` launch.